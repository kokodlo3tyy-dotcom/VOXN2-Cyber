#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <memory>
#include <cmath>
#include <array>
#include <vector>

struct MaqamPreset
{
    const char* nameEn;
    const char* nameAr;
    std::array<int, 12> intervals; // 0 = Off, 1 = Natural, 2 = Quarter-Tone Flat (1/2b)
};

static const std::array<MaqamPreset, 8> kMaqamat = {{
    { "Rast",     "راست",   {1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2} },
    { "Bayati",   "بياتي",  {1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 1, 0} },
    { "Sikah",    "سيكاه",  {2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1} },
    { "Saba",     "صبا",    {1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0} },
    { "Hijaz",    "حجاز",   {1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0} },
    { "Kurd",     "كرد",    {1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0} },
    { "Nahawand", "نهاوند", {1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1} },
    { "Ajam",     "عجم",    {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1} }
}};

static const char* kNoteNames[12] = { "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" };

struct TriggerPadData
{
    juce::String name;
    juce::String subLabel;
    int midiNote = 36;
    bool muted = false;
    bool hasCustomSample = false;
    juce::String customFileName;
    juce::AudioBuffer<float> sampleBuffer;
    double sampleRate = 44100.0;
    int playhead = -1;
    int trainedTakes = 12;
    float flashAlpha = 0.0f;
};

class VoxN2AudioProcessor : public juce::AudioProcessor
{
public:
    VoxN2AudioProcessor()
        : AudioProcessor (BusesProperties()
                          .withInput  ("Microphone Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Audio Output",     juce::AudioChannelSet::stereo(), true))
    {
        formatManager.registerBasicFormats();

        const char* defaultNames[8] = {
            "Dum Baladi", "Acoustic Snare", "Closed Hi-Hat",
            "Hand Clap", "Tak Darbuka", "Sak Shaabi",
            "Riq Shimmies", "Sub Kick 808"
        };
        const char* defaultSubs[8] = {
            "Kick / C1", "Snare / D1", "Hat / F#1",
            "Clap / D#1", "Tak / G#1", "Sak / E1",
            "Riq / A#1", "Sub / B0"
        };
        const int defaultNotes[8] = { 36, 38, 42, 39, 44, 40, 46, 35 };

        for (int i = 0; i < 8; ++i)
        {
            pads[i].name = defaultNames[i];
            pads[i].subLabel = defaultSubs[i];
            pads[i].midiNote = defaultNotes[i];
            generateBuiltInPercussion (i);
        }

        applyMaqamPreset (0, 0);
    }

    ~VoxN2AudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int) override
    {
        currentSampleRate = (sampleRate > 0.0) ? sampleRate : 44100.0;
        yinBuffer.assign (kYinWindowSize, 0.0f);
        yinWritePos = 0;
        synthPhase = 0.0;
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    bool loadCustomSampleForPad (int padIndex, const juce::File& file)
    {
        if (padIndex < 0 || padIndex >= 8 || ! file.existsAsFile())
            return false;

        std::unique_ptr<juce::AudioFormatReader> reader (formatManager.createReaderFor (file));
        if (reader == nullptr)
            return false;

        const int numSamples = (int) juce::jmin<juce::int64> (reader->lengthInSamples, (juce::int64) (reader->sampleRate * 5.0));
        juce::AudioBuffer<float> tempBuffer (1, numSamples);
        reader->read (&tempBuffer, 0, numSamples, 0, true, false);

        const juce::ScopedLock sl (sampleLock);
        pads[padIndex].sampleBuffer = std::move (tempBuffer);
        pads[padIndex].sampleRate = reader->sampleRate;
        pads[padIndex].hasCustomSample = true;
        pads[padIndex].customFileName = file.getFileNameWithoutExtension().substring (0, 12);
        pads[padIndex].playhead = -1;
        return true;
    }

    void revertPadToBuiltIn (int padIndex)
    {
        if (padIndex < 0 || padIndex >= 8) return;
        const juce::ScopedLock sl (sampleLock);
        pads[padIndex].hasCustomSample = false;
        pads[padIndex].customFileName.clear();
        generateBuiltInPercussion (padIndex);
    }

    void triggerPadNow (int padIndex)
    {
        if (padIndex < 0 || padIndex >= 8) return;
        pendingPadTrigger.store (padIndex);
    }

    void applyMaqamPreset (int root, int maqamIdx)
    {
        rootNote = juce::jlimit (0, 11, root);
        currentMaqamIndex = juce::jlimit (0, (int) kMaqamat.size() - 1, maqamIdx);

        for (int i = 0; i < 12; ++i)
        {
            int interval = (i - rootNote + 12) % 12;
            noteStates[i] = kMaqamat[currentMaqamIndex].intervals[interval];
        }
    }

    float detectPitchYIN (const std::vector<float>& buf, double sr)
    {
        const int W = (int) buf.size() / 2;
        std::vector<float> diff (W, 0.0f);
        std::vector<float> cmnd (W, 0.0f);

        for (int tau = 1; tau < W; ++tau)
        {
            float sum = 0.0f;
            for (int i = 0; i < W; ++i)
            {
                float d = buf[i] - buf[i + tau];
                sum += d * d;
            }
            diff[tau] = sum;
        }

        cmnd[0] = 1.0f;
        float runningSum = 0.0f;
        for (int tau = 1; tau < W; ++tau)
        {
            runningSum += diff[tau];
            cmnd[tau] = (runningSum > 1.0e-9f) ? (diff[tau] * (float) tau / runningSum) : 1.0f;
        }

        const int minTau = juce::jmax (2, (int) (sr / 1000.0));
        const int maxTau = juce::jmin (W - 2, (int) (sr / 65.0));
        const float threshold = 0.14f;

        int bestTau = -1;
        for (int tau = minTau; tau < maxTau; ++tau)
        {
            if (cmnd[tau] < threshold)
            {
                while (tau + 1 < maxTau && cmnd[tau + 1] < cmnd[tau])
                    ++tau;
                bestTau = tau;
                break;
            }
        }

        if (bestTau < 0)
            return -1.0f;

        float s0 = cmnd[bestTau - 1];
        float s1 = cmnd[bestTau];
        float s2 = cmnd[bestTau + 1];
        float denom = 2.0f * (2.0f * s1 - s2 - s0);
        float betterTau = (float) bestTau + ((std::abs (denom) > 1e-6f) ? (s2 - s0) / denom : 0.0f);

        return (betterTau > 0.0f) ? (float) (sr / betterTau) : -1.0f;
    }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;

        const int numSamples = buffer.getNumSamples();
        const int numInChannels = getTotalNumInputChannels();
        const int numOutChannels = getTotalNumOutputChannels();

        std::vector<float> micMono (numSamples, 0.0f);
        if (numInChannels > 0)
        {
            const float* inPtr = buffer.getReadPointer (0);
            std::copy (inPtr, inPtr + numSamples, micMono.begin());
        }

        for (int ch = 0; ch < numOutChannels; ++ch)
            buffer.clear (ch, 0, numSamples);

        float sumSq = 0.0f;
        int zeroCrossings = 0;
        for (int i = 0; i < numSamples; ++i)
        {
            float s = micMono[i];
            sumSq += s * s;
            if (i > 0 && ((micMono[i - 1] >= 0.0f && s < 0.0f) || (micMono[i - 1] < 0.0f && s >= 0.0f)))
                ++zeroCrossings;

            yinBuffer[yinWritePos] = s;
            yinWritePos = (yinWritePos + 1) % kYinWindowSize;
        }

        float rms = std::sqrt (sumSq / (float) juce::jmax (1, numSamples));
        currentInputRms.store (rms);
        float zcr = (float) zeroCrossings / (float) juce::jmax (1, numSamples);

        float envTarget = juce::jlimit (0.0f, 1.0f, rms * 8.5f);
        vowelEnv = 0.85f * vowelEnv + 0.15f * envTarget;
        if (rms > gateThreshold)
        {
            vowelEEE = juce::jlimit (0.0f, 1.0f, zcr * 3.5f);
            vowelOOO = juce::jlimit (0.0f, 1.0f, 1.0f - zcr * 2.8f);
            vowelAAA = juce::jlimit (0.0f, 1.0f, 1.0f - std::abs (zcr - 0.15f) * 4.0f);

            midiMessages.addEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, ccNumberAAA, (int) (vowelAAA * 127.0f)), 0);
            midiMessages.addEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, ccNumberEEE, (int) (vowelEEE * 127.0f)), 0);
            midiMessages.addEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, ccNumberOOO, (int) (vowelOOO * 127.0f)), 0);
            midiMessages.addEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, ccNumberENV, (int) (vowelEnv * 127.0f)), 0);
        }

        int manualPad = pendingPadTrigger.exchange (-1);
        if (manualPad >= 0 && manualPad < 8)
            fireTriggerPad (manualPad, midiMessages);

        if (triggerCooldownSamples > 0)
            triggerCooldownSamples -= numSamples;

        float onsetDelta = rms - prevRms;
        prevRms = rms;
        float onsetThresh = 0.18f - (triggerSensitivity * 0.14f);

        if (onsetDelta > onsetThresh && triggerCooldownSamples <= 0)
        {
            int detectedPad = (zcr < 0.09f) ? 0 : ((zcr < 0.20f) ? 1 : 2);
            fireTriggerPad (detectedPad, midiMessages);
            triggerCooldownSamples = (int) (currentSampleRate * 0.11);
        }

        int manualNote = guiManualMidiNote.load();
        float detectedFreq = -1.0f;

        if (manualNote >= 0)
        {
            detectedFreq = 440.0f * std::pow (2.0f, (manualNote - 69.0f) / 12.0f);
        }
        else if (rms > gateThreshold)
        {
            std::vector<float> orderedYin (kYinWindowSize);
            for (int i = 0; i < kYinWindowSize; ++i)
                orderedYin[i] = yinBuffer[(yinWritePos + i) % kYinWindowSize];

            detectedFreq = detectPitchYIN (orderedYin, currentSampleRate);
        }

        if (detectedFreq > 60.0f && detectedFreq < 1050.0f)
        {
            float rawMidi = 69.0f + 12.0f * std::log2 (detectedFreq / 440.0f);
            rawMidi += (float) (octaveShift * 12);

            int candidateMidi = (int) std::round (rawMidi);
            float hysteresisWidth = 0.48f + (stickiness * 0.45f);

            if (activeRawMidiNote >= 0 && std::abs (rawMidi - (float) activeRawMidiNote) < hysteresisWidth)
            {
                candidateMidi = activeRawMidiNote;
                confirmCounter = noteHoldFrames;
            }
            else if (candidateMidi != pendingCandidateNote)
            {
                pendingCandidateNote = candidateMidi;
                confirmCounter = 1;
            }
            else
            {
                confirmCounter++;
            }

            if (confirmCounter >= noteHoldFrames || activeRawMidiNote < 0)
            {
                activeRawMidiNote = candidateMidi;
                int snappedMidi = snapToMaqamScale (activeRawMidiNote);
                int pc = ((snappedMidi % 12) + 12) % 12;

                float targetCents = (noteStates[pc] == 2) ? quarterToneCents : 0.0f;
                if (truBendMode && manualNote < 0)
                {
                    float vocalDeviationCents = (rawMidi - (float) activeRawMidiNote) * 100.0f;
                    targetCents += vocalDeviationCents * 0.65f;
                }

                float maxBendCents = (float) pbRangeSemitones * 100.0f;
                int pbValue = 8192 + (int) std::round ((targetCents / maxBendCents) * 8192.0f);
                pbValue = juce::jlimit (0, 16383, pbValue);
                currentPitchBendValue.store (pbValue);
                currentCentsOffset.store ((int) std::round (targetCents));

                if (snappedMidi != activeOutputMidiNote)
                {
                    if (activeOutputMidiNote >= 0)
                    {
                        midiMessages.addEvent (juce::MidiMessage::noteOff (pitchMidiChannel, activeOutputMidiNote), 0);
                        stopActiveChord (midiMessages);
                    }

                    midiMessages.addEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), 0);
                    juce::uint8 vel = (juce::uint8) juce::jlimit (50, 127, (int) (vowelEnv * 115.0f + 45.0f));
                    midiMessages.addEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, vel), 0);

                    activeOutputMidiNote = snappedMidi;
                    currentDetectedMidiNote.store (snappedMidi);

                    if (chordsEnabled)
                        triggerSmartOrientalChord (snappedMidi, vel, midiMessages);
                }
                else
                {
                    midiMessages.addEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), 0);
                }
            }
        }
        else
        {
            if (activeOutputMidiNote >= 0)
            {
                midiMessages.addEvent (juce::MidiMessage::noteOff (pitchMidiChannel, activeOutputMidiNote), 0);
                midiMessages.addEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, 8192), 0);
                stopActiveChord (midiMessages);
                activeOutputMidiNote = -1;
                activeRawMidiNote = -1;
                currentDetectedMidiNote.store (-1);
                currentPitchBendValue.store (8192);
                currentCentsOffset.store (0);
            }
        }

        float* outL = buffer.getWritePointer (0);
        float* outR = (numOutChannels > 1) ? buffer.getWritePointer (1) : outL;

        if (! intellAudioMuted.load() && activeOutputMidiNote >= 0)
        {
            int pc = ((activeOutputMidiNote % 12) + 12) % 12;
            float cents = (noteStates[pc] == 2) ? quarterToneCents : 0.0f;
            double targetFreq = 440.0 * std::pow (2.0, ((double) activeOutputMidiNote - 69.0 + cents / 100.0) / 12.0);
            double phaseInc = targetFreq / currentSampleRate;

            for (int i = 0; i < numSamples; ++i)
            {
                float s = renderInternalSynthSample (synthPhase);
                synthPhase += phaseInc;
                if (synthPhase >= 1.0) synthPhase -= 1.0;

                outL[i] += s * 0.22f;
                outR[i] += s * 0.22f;
            }
        }

        if (! trigAudioMuted.load())
        {
            const juce::ScopedLock sl (sampleLock);
            for (int p = 0; p < 8; ++p)
            {
                auto& pad = pads[p];
                if (pad.playhead >= 0 && ! pad.muted && pad.sampleBuffer.getNumSamples() > 0)
                {
                    const float* samp = pad.sampleBuffer.getReadPointer (0);
                    const int totalLen = pad.sampleBuffer.getNumSamples();

                    for (int i = 0; i < numSamples; ++i)
                    {
                        if (pad.playhead < totalLen)
                        {
                            float v = samp[pad.playhead++] * 0.5f;
                            outL[i] += v;
                            outR[i] += v;
                        }
                        else
                        {
                            pad.playhead = -1;
                            break;
                        }
                    }
                }
            }
        }
    }

    int snapToMaqamScale (int midiNote) const
    {
        int pc = ((midiNote % 12) + 12) % 12;
        if (noteStates[pc] > 0)
            return midiNote;

        for (int dist = 1; dist <= 6; ++dist)
        {
            int downPc = (pc - dist + 12) % 12;
            if (noteStates[downPc] > 0)
                return midiNote - dist;

            int upPc = (pc + dist) % 12;
            if (noteStates[upPc] > 0)
                return midiNote + dist;
        }
        return midiNote;
    }

    void fireTriggerPad (int padIdx, juce::MidiBuffer& midiMessages)
    {
        auto& pad = pads[padIdx];
        pad.flashAlpha = 1.0f;
        pad.playhead = 0;

        midiMessages.addEvent (juce::MidiMessage::noteOn (triggersMidiChannel, pad.midiNote, (juce::uint8) 115), 0);
        midiMessages.addEvent (juce::MidiMessage::noteOff (triggersMidiChannel, pad.midiNote), 64);
    }

    void triggerSmartOrientalChord (int rootMidi, juce::uint8 vel, juce::MidiBuffer& midiMessages)
    {
        stopActiveChord (midiMessages);
        int pc = ((rootMidi % 12) + 12) % 12;
        bool isQuarterTone = (noteStates[pc] == 2);

        if (isQuarterTone)
        {
            activeChordNotes = { rootMidi, rootMidi + 7, rootMidi + 12 };
        }
        else
        {
            int thirdCandidate = rootMidi + 4;
            int thirdPc = ((thirdCandidate % 12) + 12) % 12;
            int thirdInterval = (noteStates[thirdPc] == 1) ? 4 : 3;
            activeChordNotes = { rootMidi, rootMidi + thirdInterval, rootMidi + 7 };
        }

        if (chordRootBassline)
            activeChordNotes.push_back (rootMidi - 12);

        for (int n : activeChordNotes)
            midiMessages.addEvent (juce::MidiMessage::noteOn (chordsMidiChannel, n, (juce::uint8) (vel * 0.78f)), 0);
    }

    void stopActiveChord (juce::MidiBuffer& midiMessages)
    {
        for (int n : activeChordNotes)
            midiMessages.addEvent (juce::MidiMessage::noteOff (chordsMidiChannel, n), 0);
        activeChordNotes.clear();
    }

    float renderInternalSynthSample (double phase) const
    {
        double twoPi = juce::MathConstants<double>::twoPi;
        switch (synthPresetIndex)
        {
            case 0: return (float) (std::sin (twoPi * phase) * 0.55 + std::sin (twoPi * phase * 2.0) * 0.30 + std::sin (twoPi * phase * 3.0) * 0.15);
            case 1: return (float) (std::sin (twoPi * phase) * 0.65 + std::sin (twoPi * phase * 3.0) * 0.25 + std::sin (twoPi * phase * 5.0) * 0.10);
            case 2: return (float) (std::sin (twoPi * phase) * 0.70 + std::sin (twoPi * phase * 2.0) * 0.20);
            case 3: return (float) (std::sin (twoPi * phase) * 0.80 + std::sin (twoPi * phase * 2.0) * 0.15);
            case 4: return (phase < 0.5) ? 0.65f : -0.65f;
            default: return (float) std::sin (twoPi * phase);
        }
    }

    void generateBuiltInPercussion (int padIndex)
    {
        const int len = (int) (44100 * 0.22);
        pads[padIndex].sampleBuffer.setSize (1, len);
        float* out = pads[padIndex].sampleBuffer.getWritePointer (0);
        juce::Random rng (padIndex * 101 + 7);

        for (int i = 0; i < len; ++i)
        {
            float t = (float) i / 44100.0f;
            float env = std::exp (-t * (14.0f + padIndex * 3.0f));
            float freq = (padIndex == 0 || padIndex == 7) ? (130.0f * std::exp (-t * 22.0f) + 48.0f)
                                                          : (240.0f + padIndex * 95.0f);
            float tone = std::sin (juce::MathConstants<float>::twoPi * freq * t);
            float noise = (rng.nextFloat() * 2.0f - 1.0f) * ((padIndex == 0) ? 0.12f : 0.55f);
            out[i] = (tone + noise) * env;
        }
    }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "VOXN 2 Cyber Gold"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override
    {
        juce::ValueTree state ("VOXN2State");
        state.setProperty ("rootNote", rootNote, nullptr);
        state.setProperty ("maqamIndex", currentMaqamIndex, nullptr);
        state.setProperty ("quarterToneCents", quarterToneCents, nullptr);
        state.setProperty ("stickiness", stickiness, nullptr);
        state.setProperty ("intellMuted", intellAudioMuted.load(), nullptr);
        state.setProperty ("trigMuted", trigAudioMuted.load(), nullptr);
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
        if (xmlState != nullptr)
        {
            auto state = juce::ValueTree::fromXml (*xmlState);
            rootNote = state.getProperty ("rootNote", 0);
            currentMaqamIndex = state.getProperty ("maqamIndex", 0);
            quarterToneCents = state.getProperty ("quarterToneCents", -50.0f);
            stickiness = state.getProperty ("stickiness", 0.65f);
            intellAudioMuted.store ((bool) state.getProperty ("intellMuted", false));
            trigAudioMuted.store ((bool) state.getProperty ("trigMuted", false));
            applyMaqamPreset (rootNote, currentMaqamIndex);
        }
    }

    std::array<TriggerPadData, 8> pads;
    std::array<int, 12> noteStates { 1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2 };

    int rootNote = 0;
    int currentMaqamIndex = 0;
    float quarterToneCents = -50.0f;
    float stickiness = 0.65f;
    float gateThreshold = 0.018f;
    float triggerSensitivity = 0.72f;
    int noteHoldFrames = 2;
    int pbRangeSemitones = 2;
    int octaveShift = 0;
    bool chordsEnabled = false;
    bool chordRootBassline = true;
    bool truBendMode = false;
    int synthPresetIndex = 0;

    int pitchMidiChannel = 1;
    int chordsMidiChannel = 2;
    int triggersMidiChannel = 10;

    int ccNumberAAA = 74;
    int ccNumberEEE = 71;
    int ccNumberOOO = 1;
    int ccNumberENV = 11;

    float vowelAAA = 0.45f;
    float vowelEEE = 0.30f;
    float vowelOOO = 0.60f;
    float vowelEnv = 0.50f;

    std::atomic<bool> intellAudioMuted { false };
    std::atomic<bool> trigAudioMuted   { false };
    std::atomic<int> currentDetectedMidiNote { -1 };
    std::atomic<int> currentPitchBendValue { 8192 };
    std::atomic<int> currentCentsOffset { 0 };
    std::atomic<float> currentInputRms { 0.0f };
    std::atomic<int> guiManualMidiNote { -1 };
    std::atomic<int> pendingPadTrigger { -1 };

private:
    static constexpr int kYinWindowSize = 1024;
    std::vector<float> yinBuffer;
    int yinWritePos = 0;
    double currentSampleRate = 44100.0;
    double synthPhase = 0.0;

    int activeRawMidiNote = -1;
    int activeOutputMidiNote = -1;
    int pendingCandidateNote = -1;
    int confirmCounter = 0;
    float prevRms = 0.0f;
    int triggerCooldownSamples = 0;

    std::vector<int> activeChordNotes;
    juce::AudioFormatManager formatManager;
    juce::CriticalSection sampleLock;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxN2AudioProcessor)
};

class VoxN2AudioProcessorEditor : public juce::AudioProcessorEditor,
                                  public juce::Timer,
                                  public juce::FileDragAndDropTarget
{
public:
    explicit VoxN2AudioProcessorEditor (VoxN2AudioProcessor& p)
        : AudioProcessorEditor (&p), proc (p)
    {
        setSize (960, 600);
        startTimerHz (30);
    }

    ~VoxN2AudioProcessorEditor() override
    {
        stopTimer();
    }

    void timerCallback() override
    {
        for (auto& pad : proc.pads)
            if (pad.flashAlpha > 0.01f)
                pad.flashAlpha *= 0.82f;

        if (taqsimAutoPlaying)
        {
            taqsimStepCounter++;
            if (taqsimStepCounter % 12 == 0)
            {
                static const int phrase[] = { 60, 62, 64, 65, 67, 65, 64, 62, 60, 71, 60 };
                int idx = (taqsimStepCounter / 12) % 11;
                proc.guiManualMidiNote.store (phrase[idx]);
            }
        }
        repaint();
    }

    bool isInterestedInFileDrag (const juce::StringArray& files) override
    {
        for (auto& f : files)
            if (f.endsWithIgnoreCase (".wav") || f.endsWithIgnoreCase (".mp3") || f.endsWithIgnoreCase (".flac"))
                return true;
        return false;
    }

    void filesDropped (const juce::StringArray& files, int x, int y) override
    {
        if (files.isEmpty()) return;
        juce::File droppedFile (files[0]);

        for (int i = 0; i < 8; ++i)
        {
            if (padBounds[i].contains ((float) x, (float) y))
            {
                proc.loadCustomSampleForPad (i, droppedFile);
                selectedTrainPad = i;
                repaint();
                return;
            }
        }
    }

    void openSampleChooserForPad (int padIndex)
    {
        fileChooser = std::make_unique<juce::FileChooser> (
            "Select Custom Kick / Snare / Oriental Percussion (.wav, .mp3)",
            juce::File{},
            "*.wav;*.mp3;*.flac;*.ogg");

        fileChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this, padIndex] (const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file.existsAsFile())
                {
                    proc.loadCustomSampleForPad (padIndex, file);
                    repaint();
                }
            });
    }

    void paint (juce::Graphics& g) override
    {
        const juce::Colour bgDark     (0xff1b1b1f);
        const juce::Colour cardDark   (0xff25252b);
        const juce::Colour cyberGold  (0xffe5b84b);
        const juce::Colour neonPink   (0xfff03e76);
        const juce::Colour textBright (0xfff4f0e8);
        const juce::Colour textMuted  (0xff8e8d96);

        g.fillAll (bgDark);

        g.setColour (juce::Colour (0xff141417));
        g.fillRect (0, 0, getWidth(), 42);

        g.setColour (cyberGold);
        g.drawEllipse (14.0f, 10.0f, 22.0f, 22.0f, 2.0f);
        g.setFont (juce::Font (15.0f, juce::Font::bold));
        g.setColour (textBright);
        g.drawText ("VOXN 2", 42, 8, 65, 26, juce::Justification::centredLeft);

        g.setColour (cyberGold);
        g.setFont (juce::Font (12.0f, juce::Font::bold));
        g.drawText ("CYBER GOLD", 108, 8, 90, 26, juce::Justification::centredLeft);

        taqsimButtonRect = { 710.0f, 8.0f, 130.0f, 26.0f };
        g.setColour (taqsimAutoPlaying ? neonPink : cyberGold);
        g.fillRoundedRectangle (taqsimButtonRect, 6.0f);
        g.setColour (juce::Colours::black);
        g.setFont (juce::Font (12.0f, juce::Font::bold));
        g.drawText (taqsimAutoPlaying ? "STOP TAQSIM" : "CYFER TAQSIM", taqsimButtonRect, juce::Justification::centred);

        juce::Rectangle<float> micBadge (848.0f, 8.0f, 96.0f, 26.0f);
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (micBadge, 6.0f);
        g.setColour (cyberGold.withAlpha (0.6f));
        g.drawRoundedRectangle (micBadge, 6.0f, 1.0f);
        g.setColour (textBright);
        g.setFont (11.5f);
        g.drawText ("MIC LIVE", micBadge, juce::Justification::centred);

        const char* tabTitles[5] = { "Play", "Train (Kit)", "Key (Maqam)", "Chords", "Assign (MIDI)" };
        for (int i = 0; i < 5; ++i)
        {
            tabRects[i] = { 160.0f + i * 128.0f, 46.0f, 122.0f, 30.0f };
            bool isCurr = (currentTab == i);
            g.setColour (isCurr ? cardDark : juce::Colour (0xff19191e));
            g.fillRoundedRectangle (tabRects[i], 6.0f);
            g.setColour (isCurr ? cyberGold : textMuted.withAlpha (0.3f));
            g.drawRoundedRectangle (tabRects[i], 6.0f, isCurr ? 1.5f : 1.0f);
            g.setColour (isCurr ? cyberGold : textMuted);
            g.setFont (juce::Font (12.5f, juce::Font::bold));
            g.drawText (tabTitles[i], tabRects[i], juce::Justification::centred);
        }

        juce::Rectangle<float> mainCard (28.0f, 74.0f, 920.0f, 336.0f);
        g.setColour (cardDark);
        g.fillRoundedRectangle (mainCard, 10.0f);
        g.setColour (cyberGold.withAlpha (0.28f));
        g.drawRoundedRectangle (mainCard, 10.0f, 1.2f);

        float rmsNorm = juce::jlimit (0.0f, 1.0f, proc.currentInputRms.load() * 10.0f);
        for (int d = 0; d < 28; ++d)
        {
            float dy = 395.0f - d * 11.0f;
            bool lit = ((float) d / 28.0f) <= rmsNorm;
            g.setColour (lit ? (d > 20 ? neonPink : cyberGold) : textMuted.withAlpha (0.25f));
            g.fillEllipse (11.0f, dy, 4.0f, 4.0f);
        }

        if      (currentTab == 0) drawPlayTab   (g, cyberGold, neonPink, textBright, textMuted);
        else if (currentTab == 1) drawTrainTab  (g, cyberGold, neonPink, textBright, textMuted);
        else if (currentTab == 2) drawKeyTab    (g, cyberGold, neonPink, textBright, textMuted);
        else if (currentTab == 3) drawChordsTab (g, cyberGold, neonPink, textBright, textMuted);
        else if (currentTab == 4) drawAssignTab (g, cyberGold, neonPink, textBright, textMuted);

        drawBottomSection (g, cyberGold, neonPink, textBright, textMuted);
    }

    void drawPlayTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::Font (12.5f, juce::Font::bold));
        g.drawText ("TRIGGERS (DRAG WAV/MP3 OR CLICK [WAV])", 44, 82, 280, 20, juce::Justification::centredLeft);

        for (int i = 0; i < 9; ++i)
        {
            int row = i / 3;
            int col = i % 3;
            juce::Rectangle<float> r (44.0f + col * 92.0f, 108.0f + row * 96.0f, 84.0f, 88.0f);

            if (i < 8)
            {
                padBounds[i] = r;
                auto& pad = proc.pads[i];
                juce::Colour borderCol = (row == 0) ? pink : gold;

                g.setColour (juce::Colour (0xff1d1d22).interpolatedWith (borderCol, pad.flashAlpha * 0.45f));
                g.fillRoundedRectangle (r, 7.0f);
                g.setColour (borderCol.withAlpha (0.75f + pad.flashAlpha * 0.25f));
                g.drawRoundedRectangle (r, 7.0f, 1.4f);

                padLoadRects[i] = { r.getX() + 5.0f, r.getY() + 5.0f, 24.0f, 16.0f };
                g.setColour (gold.withAlpha (0.22f));
                g.fillRoundedRectangle (padLoadRects[i], 3.0f);
                g.setColour (gold);
                g.setFont (9.5f);
                g.drawText ("WAV", padLoadRects[i], juce::Justification::centred);

                padMuteRects[i] = { r.getRight() - 29.0f, r.getY() + 5.0f, 24.0f, 16.0f };
                g.setColour (pad.muted ? pink : juce::Colour (0xff32323a));
                g.fillRoundedRectangle (padMuteRects[i], 3.0f);
                g.setColour (bright);
                g.drawText (pad.muted ? "M" : "ON", padMuteRects[i], juce::Justification::centred);

                g.setColour (bright);
                g.setFont (juce::Font (11.0f, juce::Font::bold));
                g.drawText (pad.name, (int) r.getX() + 5, (int) r.getY() + 42, (int) r.getWidth() - 10, 18, juce::Justification::centredLeft);

                g.setColour (pad.hasCustomSample ? gold : muted);
                g.setFont (10.0f);
                g.drawText (pad.hasCustomSample ? ("[" + pad.customFileName + "]") : pad.subLabel,
                            (int) r.getX() + 5, (int) r.getY() + 62, (int) r.getWidth() - 10, 16, juce::Justification::centredLeft);
            }
            else
            {
                g.setColour (juce::Colour (0xff1d1d22));
                g.fillRoundedRectangle (r, 7.0f);
                g.setColour (gold.withAlpha (0.35f));
                g.drawRoundedRectangle (r, 7.0f, 1.0f);
                g.setFont (20.0f);
                g.drawText ("+", r, juce::Justification::centred);
            }
        }

        drawPitchRadarWheel (g, 488.0f, 248.0f, 122.0f, gold, pink, bright, muted);

        g.setColour (muted);
        g.setFont (11.5f);
        g.drawText ("Stickiness (Hysteresis):", 635, 82, 130, 20, juce::Justification::centredLeft);
        stickinessSliderRect = { 768.0f, 86.0f, 155.0f, 12.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (stickinessSliderRect, 6.0f);
        g.setColour (pink);
        g.fillRoundedRectangle (stickinessSliderRect.getX(), stickinessSliderRect.getY(),
                                stickinessSliderRect.getWidth() * proc.stickiness, 12.0f, 6.0f);

        const float cx = 795.0f, cy = 248.0f, offset = 64.0f;
        struct VowelNode { const char* label; float x; float y; float val; juce::Colour col; };
        VowelNode nodes[4] = {
            { "aaa", cx,          cy - offset, proc.vowelAAA, gold },
            { "eee", cx - offset, cy,          proc.vowelEEE, pink },
            { "ooo", cx + offset, cy,          proc.vowelOOO, pink },
            { "env", cx,          cy + offset, proc.vowelEnv, gold }
        };

        for (int i = 0; i < 4; ++i)
        {
            vowelRects[i] = { nodes[i].x - 28.0f, nodes[i].y - 28.0f, 56.0f, 56.0f };
            g.setColour (juce::Colour (0xff19191f));
            g.fillEllipse (vowelRects[i]);
            g.setColour (nodes[i].col.withAlpha (0.35f));
            g.drawEllipse (vowelRects[i], 2.0f);

            juce::Path arc;
            arc.addCentredArc (nodes[i].x, nodes[i].y, 28.0f, 28.0f, 0.0f,
                               -juce::MathConstants<float>::halfPi,
                               -juce::MathConstants<float>::halfPi + nodes[i].val * juce::MathConstants<float>::twoPi, true);
            g.setColour (nodes[i].col);
            g.strokePath (arc, juce::PathStrokeType (3.2f));

            g.setColour (bright);
            g.setFont (juce::Font (12.0f, juce::Font::bold));
            g.drawText (nodes[i].label, vowelRects[i], juce::Justification::centred);
        }
    }

    void drawPitchRadarWheel (juce::Graphics& g, float cx, float cy, float radius,
                              juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour)
    {
        int activeMidi = proc.currentDetectedMidiNote.load();
        int activePc = (activeMidi >= 0) ? ((activeMidi % 12) + 12) % 12 : -1;

        for (int i = 0; i < 12; ++i)
        {
            float angleDeg = (float) i * 30.0f - 90.0f;
            float a1 = juce::degreesToRadians (angleDeg - 13.0f);
            float a2 = juce::degreesToRadians (angleDeg + 13.0f);

            int state = proc.noteStates[i];
            bool isPlaying = (i == activePc);

            juce::Path seg;
            seg.addPieSegment (cx - radius, cy - radius, radius * 2.0f, radius * 2.0f,
                               a1 + juce::MathConstants<float>::halfPi,
                               a2 + juce::MathConstants<float>::halfPi, 0.56f);

            if (state == 0)
                g.setColour (juce::Colour (0xff18181c));
            else if (isPlaying)
                g.setColour (state == 2 ? gold : pink);
            else if (state == 2)
                g.setColour (gold.withAlpha (0.28f));
            else
                g.setColour (juce::Colour (0xff32323b));

            g.fillPath (seg);
            g.setColour (state == 2 ? gold : pink.withAlpha (0.35f));
            g.strokePath (seg, juce::PathStrokeType (1.2f));

            float midRad = juce::degreesToRadians (angleDeg);
            float tx = cx + std::cos (midRad) * (radius * 0.78f);
            float ty = cy + std::sin (midRad) * (radius * 0.78f);
            wheelSegmentCenters[i] = { tx, ty };

            if (state > 0)
            {
                g.setColour (isPlaying ? juce::Colours::black : bright);
                g.setFont (juce::Font (11.5f, juce::Font::bold));
                juce::String label = kNoteNames[i];
                if (state == 2) label += " 1/2b";
                g.drawText (label, (int) tx - 22, (int) ty - 10, 44, 20, juce::Justification::centred);
            }
        }

        g.setColour (juce::Colour (0xff19191f));
        g.fillEllipse (cx - 48.0f, cy - 48.0f, 96.0f, 96.0f);
        g.setColour (gold);
        g.drawEllipse (cx - 48.0f, cy - 48.0f, 96.0f, 96.0f, 2.2f);

        juce::String centerNote = (activeMidi >= 0)
            ? (juce::String (kNoteNames[activePc]) + juce::String (activeMidi / 12 - 1))
            : "READY";
        g.setColour (bright);
        g.setFont (juce::Font (16.0f, juce::Font::bold));
        g.drawText (centerNote, (int) cx - 40, (int) cy - 18, 80, 22, juce::Justification::centred);

        g.setColour (gold);
        g.setFont (11.0f);
        int cents = proc.currentCentsOffset.load();
        g.drawText (juce::String (cents) + " cents", (int) cx - 40, (int) cy + 4, 80, 18, juce::Justification::centred);
    }

    void drawTrainTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText ("SELECT PAD TO TRAIN OR LOAD CUSTOM WAV/MP3:", 48, 88, 360, 22, juce::Justification::centredLeft);

        for (int i = 0; i < 8; ++i)
        {
            juce::Rectangle<float> r (48.0f + (i % 2) * 145.0f, 118.0f + (i / 2) * 65.0f, 135.0f, 54.0f);
            padBounds[i] = r;
            bool sel = (selectedTrainPad == i);
            g.setColour (sel ? gold.withAlpha (0.22f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (r, 6.0f);
            g.setColour (sel ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (r, 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::Font (12.0f, juce::Font::bold));
            g.drawText (proc.pads[i].name, r.reduced (8.0f), juce::Justification::topLeft);
            g.setColour (proc.pads[i].hasCustomSample ? gold : muted);
            g.setFont (10.5f);
            g.drawText (proc.pads[i].hasCustomSample ? proc.pads[i].customFileName : proc.pads[i].subLabel,
                        r.reduced (8.0f), juce::Justification::bottomLeft);
        }

        float cx = 520.0f, cy = 245.0f, rad = 95.0f;
        for (int t = 0; t < 12; ++t)
        {
            float a = juce::degreesToRadians ((float) t * 30.0f - 90.0f);
            float tx = cx + std::cos (a) * rad;
            float ty = cy + std::sin (a) * rad;
            g.setColour (t < proc.pads[selectedTrainPad].trainedTakes ? gold : juce::Colour (0xff18181d));
            g.fillEllipse (tx - 11.0f, ty - 11.0f, 22.0f, 22.0f);
        }

        g.setColour (bright);
        g.setFont (juce::Font (15.0f, juce::Font::bold));
        g.drawText ("12 / 12 TAKES", (int) cx - 65, (int) cy - 12, 130, 24, juce::Justification::centred);

        trainLoadButtonRect   = { 690.0f, 150.0f, 220.0f, 38.0f };
        trainPlayButtonRect   = { 690.0f, 202.0f, 220.0f, 38.0f };
        trainRevertButtonRect = { 690.0f, 254.0f, 220.0f, 38.0f };

        g.setColour (gold);
        g.fillRoundedRectangle (trainLoadButtonRect, 6.0f);
        g.setColour (juce::Colours::black);
        g.setFont (juce::Font (12.5f, juce::Font::bold));
        g.drawText ("LOAD CUSTOM WAV / MP3", trainLoadButtonRect, juce::Justification::centred);

        g.setColour (pink);
        g.fillRoundedRectangle (trainPlayButtonRect, 6.0f);
        g.setColour (bright);
        g.drawText ("AUDITION / PLAY PAD", trainPlayButtonRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff32323b));
        g.fillRoundedRectangle (trainRevertButtonRect, 6.0f);
        g.setColour (bright);
        g.drawText ("REVERT TO BUILT-IN SOUND", trainRevertButtonRect, juce::Justification::centred);
    }

    void drawKeyTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour muted)
    {
        drawPitchRadarWheel (g, 220.0f, 242.0f, 118.0f, gold, pink, bright, muted);

        g.setColour (gold);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText ("ORIENTAL MAQAM PRESETS & QUARTER-TONE DEPTH (1/2b):", 395, 86, 480, 22, juce::Justification::centredLeft);

        for (int m = 0; m < 8; ++m)
        {
            maqamPresetRects[m] = { 395.0f + (m % 2) * 255.0f, 118.0f + (m / 2) * 52.0f, 240.0f, 42.0f };
            bool active = (proc.currentMaqamIndex == m);
            g.setColour (active ? gold.withAlpha (0.25f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (maqamPresetRects[m], 6.0f);
            g.setColour (active ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (maqamPresetRects[m], 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::Font (13.0f, juce::Font::bold));
            g.drawText (juce::String (kMaqamat[m].nameEn) + "  (" + juce::String::fromUTF8 (kMaqamat[m].nameAr) + ")",
                        maqamPresetRects[m].reduced (12.0f), juce::Justification::centredLeft);
        }

        g.setColour (bright);
        g.setFont (12.5f);
        g.drawText ("Quarter-Tone Depth: " + juce::String ((int) proc.quarterToneCents) + " cents",
                    395, 342, 240, 22, juce::Justification::centredLeft);

        qtDepthSliderRect = { 625.0f, 346.0f, 265.0f, 14.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (qtDepthSliderRect, 7.0f);
        float normQt = (proc.quarterToneCents - (-65.0f)) / 30.0f;
        g.setColour (gold);
        g.fillRoundedRectangle (qtDepthSliderRect.getX(), qtDepthSliderRect.getY(),
                                qtDepthSliderRect.getWidth() * juce::jlimit (0.0f, 1.0f, normQt), 14.0f, 7.0f);
    }

    void drawChordsTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.5f, juce::Font::bold));
        g.drawText ("SMART ORIENTAL CHORDS (AUTO 5TH POWER ON QUARTER-TONE DEGREES):", 48, 88, 620, 22, juce::Justification::centredLeft);

        chordsToggleRect = { 48.0f, 124.0f, 210.0f, 40.0f };
        g.setColour (proc.chordsEnabled ? pink : juce::Colour (0xff32323b));
        g.fillRoundedRectangle (chordsToggleRect, 7.0f);
        g.setColour (bright);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText (proc.chordsEnabled ? "CHORDS: ENABLED" : "CHORDS: OFF (CLICK)", chordsToggleRect, juce::Justification::centred);

        const char* roman[7] = { "I", "II", "III", "IV", "V", "VI", "VII" };
        int degIdx = 0;
        for (int pc = 0; pc < 12 && degIdx < 7; ++pc)
        {
            int note = (proc.rootNote + pc) % 12;
            if (proc.noteStates[note] > 0)
            {
                juce::Rectangle<float> col (48.0f + degIdx * 124.0f, 182.0f, 114.0f, 195.0f);
                bool isQt = (proc.noteStates[note] == 2);
                g.setColour (juce::Colour (0xff1b1b20));
                g.fillRoundedRectangle (col, 7.0f);
                g.setColour (isQt ? gold : pink.withAlpha (0.5f));
                g.drawRoundedRectangle (col, 7.0f, 1.4f);

                g.setColour (bright);
                g.setFont (juce::Font (14.0f, juce::Font::bold));
                g.drawText (juce::String (roman[degIdx]) + " - " + kNoteNames[note], col.removeFromTop (36.0f), juce::Justification::centred);

                g.setColour (isQt ? gold : bright);
                g.setFont (12.0f);
                g.drawText (isQt ? "5th Power\n(Quarter-Safe)" : "Maqam Triad\n+ Bassline", col, juce::Justification::centred);
                ++degIdx;
            }
        }
    }

    void drawAssignTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.5f, juce::Font::bold));
        g.drawText ("MIDI ROUTING & PITCH TRACKING MODE (FL STUDIO / DAW):", 48, 88, 520, 22, juce::Justification::centredLeft);

        bendModeRect = { 48.0f, 130.0f, 260.0f, 44.0f };
        g.setColour (proc.truBendMode ? pink : gold);
        g.fillRoundedRectangle (bendModeRect, 7.0f);
        g.setColour (juce::Colours::black);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText (proc.truBendMode ? "MODE: TruBend (Mawal)" : "MODE: IntelliBend (Maqam Lock)", bendModeRect, juce::Justification::centred);

        g.setColour (bright);
        g.setFont (13.0f);
        g.drawText ("Pitch + Quarter-Tone MIDI Channel: Ch " + juce::String (proc.pitchMidiChannel)
                    + "   |   Chords Channel: Ch " + juce::String (proc.chordsMidiChannel)
                    + "   |   Triggers Channel: Ch " + juce::String (proc.triggersMidiChannel),
                    48, 198, 700, 26, juce::Justification::centredLeft);

        g.setColour (muted);
        g.drawText ("Vowel CC Mapping:   AAA -> CC 74 (Cutoff)   |   EEE -> CC 71 (Resonance)   |   OOO -> CC 1 (Mod)   |   ENV -> CC 11 (Expression)",
                    48, 236, 820, 26, juce::Justification::centredLeft);
    }

    void drawBottomSection (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour)
    {
        g.setColour (bright);
        g.setFont (juce::Font (12.0f, juce::Font::bold));
        g.drawText ("PB: +/-" + juce::String (proc.pbRangeSemitones), 120, 416, 70, 22, juce::Justification::centredLeft);

        rootCycleRect = { 210.0f, 415.0f, 74.0f, 24.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (rootCycleRect, 5.0f);
        g.setColour (gold);
        g.drawText ("Key: " + juce::String (kNoteNames[proc.rootNote]), rootCycleRect, juce::Justification::centred);

        maqamCycleRect = { 292.0f, 415.0f, 140.0f, 24.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (maqamCycleRect, 5.0f);
        g.setColour (gold);
        g.drawText (kMaqamat[proc.currentMaqamIndex].nameEn, maqamCycleRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (38.0f, 446.0f, 24.0f, 102.0f, 5.0f);
        g.fillRoundedRectangle (70.0f, 446.0f, 24.0f, 102.0f, 5.0f);

        float pbNorm = (float) proc.currentPitchBendValue.load() / 16383.0f;
        float pbY = 446.0f + (1.0f - pbNorm) * 86.0f;
        g.setColour (pink);
        g.fillRoundedRectangle (40.0f, pbY, 20.0f, 14.0f, 3.0f);

        float modY = 446.0f + (1.0f - proc.vowelEnv) * 86.0f;
        g.setColour (gold);
        g.fillRoundedRectangle (72.0f, modY, 20.0f, 14.0f, 3.0f);

        const float kbX = 114.0f, kbY = 446.0f, kbW = 830.0f, kbH = 102.0f;
        const int numKeys = 36;
        const float keyW = kbW / (float) numKeys;
        int activeMidi = proc.currentDetectedMidiNote.load();

        for (int k = 0; k < numKeys; ++k)
        {
            int midi = 48 + k;
            int pc = midi % 12;
            int state = proc.noteStates[pc];
            bool isPlaying = (midi == activeMidi);

            juce::Rectangle<float> kr (kbX + k * keyW, kbY, keyW - 1.5f, kbH);
            pianoKeyRects[k] = kr;

            if (isPlaying)
                g.setColour (bright);
            else if (state == 2)
                g.setColour (gold);
            else if (state == 1)
                g.setColour (pink.interpolatedWith (gold, (float) k / (float) numKeys * 0.45f));
            else
                g.setColour (juce::Colour (0xff232329));

            g.fillRoundedRectangle (kr, 3.0f);

            if (state == 2)
            {
                g.setColour (juce::Colours::black);
                g.setFont (juce::Font (9.0f, juce::Font::bold));
                g.drawText ("1/2b", kr.removeFromBottom (20.0f), juce::Justification::centred);
            }
        }

        synthCycleRect = { 340.0f, 560.0f, 195.0f, 28.0f };
        g.setColour (juce::Colour (0xff25252c));
        g.fillRoundedRectangle (synthCycleRect, 5.0f);
        g.setColour (gold);
        const char* synthNames[6] = { "Synth: Electric Saz", "Synth: G-Clarinet", "Synth: Electric Oud",
                                      "Synth: Kawala", "Synth: Mizmar", "Synth: Sub Bass" };
        g.drawText (synthNames[proc.synthPresetIndex % 6], synthCycleRect, juce::Justification::centred);

        intellMuteRect = { 555.0f, 560.0f, 130.0f, 28.0f };
        bool intMuted = proc.intellAudioMuted.load();
        g.setColour (intMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (intellMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (intMuted ? "Intell Audio: MUTED" : "Intell Audio: ON", intellMuteRect, juce::Justification::centred);

        trigMuteRect = { 695.0f, 560.0f, 125.0f, 28.0f };
        bool trMuted = proc.trigAudioMuted.load();
        g.setColour (trMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (trigMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (trMuted ? "Trig Audio: MUTED" : "Trig Audio: ON", trigMuteRect, juce::Justification::centred);

        g.setColour (gold);
        g.setFont (juce::Font (12.0f, juce::Font::bold));
        g.drawText ("MIDI OUT: ACTIVE", 830, 560, 118, 28, juce::Justification::centred);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        juce::Point<float> pt = e.position;

        if (taqsimButtonRect.contains (pt))
        {
            taqsimAutoPlaying = ! taqsimAutoPlaying;
            if (! taqsimAutoPlaying) proc.guiManualMidiNote.store (-1);
            repaint();
            return;
        }

        for (int i = 0; i < 5; ++i)
        {
            if (tabRects[i].contains (pt))
            {
                currentTab = i;
                repaint();
                return;
            }
        }

        if (intellMuteRect.contains (pt))
        {
            proc.intellAudioMuted.store (! proc.intellAudioMuted.load());
            repaint();
            return;
        }

        if (trigMuteRect.contains (pt))
        {
            proc.trigAudioMuted.store (! proc.trigAudioMuted.load());
            repaint();
            return;
        }

        if (synthCycleRect.contains (pt))
        {
            proc.synthPresetIndex = (proc.synthPresetIndex + 1) % 6;
            repaint();
            return;
        }

        if (rootCycleRect.contains (pt))
        {
            proc.applyMaqamPreset ((proc.rootNote + 1) % 12, proc.currentMaqamIndex);
            repaint();
            return;
        }

        if (maqamCycleRect.contains (pt))
        {
            proc.applyMaqamPreset (proc.rootNote, (proc.currentMaqamIndex + 1) % (int) kMaqamat.size());
            repaint();
            return;
        }

        for (int k = 0; k < 36; ++k)
        {
            if (pianoKeyRects[k].contains (pt))
            {
                proc.guiManualMidiNote.store (48 + k);
                repaint();
                return;
            }
        }

        if (currentTab == 0)
        {
            for (int i = 0; i < 8; ++i)
            {
                if (padLoadRects[i].contains (pt)) { openSampleChooserForPad (i); return; }
                if (padMuteRects[i].contains (pt)) { proc.pads[i].muted = ! proc.pads[i].muted; repaint(); return; }
                if (padBounds[i].contains (pt))    { proc.triggerPadNow (i); repaint(); return; }
            }

            if (stickinessSliderRect.contains (pt))
            {
                proc.stickiness = juce::jlimit (0.0f, 1.0f, (pt.x - stickinessSliderRect.getX()) / stickinessSliderRect.getWidth());
                repaint();
                return;
            }

            for (int i = 0; i < 12; ++i)
            {
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 24.0f)
                {
                    if (e.mods.isRightButtonDown() || e.mods.isShiftDown())
                        proc.noteStates[i] = (proc.noteStates[i] + 1) % 3;
                    else
                        proc.guiManualMidiNote.store (60 + i);
                    repaint();
                    return;
                }
            }
        }
        else if (currentTab == 1)
        {
            for (int i = 0; i < 8; ++i)
                if (padBounds[i].contains (pt)) { selectedTrainPad = i; repaint(); return; }

            if (trainLoadButtonRect.contains (pt))   { openSampleChooserForPad (selectedTrainPad); return; }
            if (trainPlayButtonRect.contains (pt))   { proc.triggerPadNow (selectedTrainPad); return; }
            if (trainRevertButtonRect.contains (pt)) { proc.revertPadToBuiltIn (selectedTrainPad); repaint(); return; }
        }
        else if (currentTab == 2)
        {
            for (int m = 0; m < 8; ++m)
                if (maqamPresetRects[m].contains (pt)) { proc.applyMaqamPreset (proc.rootNote, m); repaint(); return; }

            if (qtDepthSliderRect.contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - qtDepthSliderRect.getX()) / qtDepthSliderRect.getWidth());
                proc.quarterToneCents = -65.0f + norm * 30.0f;
                repaint();
                return;
            }

            for (int i = 0; i < 12; ++i)
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 24.0f)
                {
                    proc.noteStates[i] = (proc.noteStates[i] + 1) % 3;
                    repaint();
                    return;
                }
        }
        else if (currentTab == 3)
        {
            if (chordsToggleRect.contains (pt)) { proc.chordsEnabled = ! proc.chordsEnabled; repaint(); return; }
        }
        else if (currentTab == 4)
        {
            if (bendModeRect.contains (pt)) { proc.truBendMode = ! proc.truBendMode; repaint(); return; }
        }
    }

    void mouseUp (const juce::MouseEvent&) override
    {
        if (! taqsimAutoPlaying)
            proc.guiManualMidiNote.store (-1);
    }

private:
    VoxN2AudioProcessor& proc;
    int currentTab = 0;
    int selectedTrainPad = 0;
    bool taqsimAutoPlaying = false;
    int taqsimStepCounter = 0;

    std::array<juce::Rectangle<float>, 5> tabRects;
    std::array<juce::Rectangle<float>, 8> padBounds;
    std::array<juce::Rectangle<float>, 8> padLoadRects;
    std::array<juce::Rectangle<float>, 8> padMuteRects;
    std::array<juce::Point<float>, 12> wheelSegmentCenters;
    std::array<juce::Rectangle<float>, 4> vowelRects;
    std::array<juce::Rectangle<float>, 8> maqamPresetRects;
    std::array<juce::Rectangle<float>, 36> pianoKeyRects;

    juce::Rectangle<float> taqsimButtonRect, stickinessSliderRect, qtDepthSliderRect;
    juce::Rectangle<float> trainLoadButtonRect, trainPlayButtonRect, trainRevertButtonRect;
    juce::Rectangle<float> chordsToggleRect, bendModeRect;
    juce::Rectangle<float> rootCycleRect, maqamCycleRect, synthCycleRect, intellMuteRect, trigMuteRect;

    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (VoxN2AudioProcessorEditor)
};

juce::AudioProcessorEditor* VoxN2AudioProcessor::createEditor()
{
    return new VoxN2AudioProcessorEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new VoxN2AudioProcessor();
}
