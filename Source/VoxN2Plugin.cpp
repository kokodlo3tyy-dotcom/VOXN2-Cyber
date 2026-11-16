#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <memory>
#include <cmath>
#include <array>
#include <vector>

enum EngineExclusiveMode
{
    ModeMelodyOnly   = 0,
    ModeTriggersOnly = 1,
    ModeChordsOnly   = 2
};

enum VocalRangePreset
{
    RangeFollow   = 0,
    RangeBass     = 1,
    RangeBaritone = 2,
    RangeTenor    = 3,
    RangeAlto     = 4,
    RangeSoprano  = 5
};

struct MaqamPreset
{
    const char* nameEn;
    const char* descEn;
    std::array<int, 12> intervals;
};

static const std::array<MaqamPreset, 10> kMaqamat = {{
    { "Major",    "Ionian Natural",     {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1} },
    { "Minor",    "Aeolian Natural",    {1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0} },
    { "Rast",     "1/2b III & VII",     {1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2} },
    { "Bayati",   "1/2b II & bVII",     {1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 1, 0} },
    { "Sikah",    "1/2b I & V",         {2, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 1} },
    { "Saba",     "1/2b II & bIV",      {1, 0, 2, 1, 0, 0, 1, 0, 1, 0, 1, 0} },
    { "Hijaz",    "Oriental Aug 2nd",   {1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0} },
    { "Kurd",     "Phrygian Oriental",  {1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0} },
    { "Nahawand", "Harmonic Oriental",  {1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1} },
    { "Ajam",     "Oriental Major",     {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1} }
}};

static const char* kNoteNames[12] = { "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" };

struct ChordVoicingOption
{
    juce::String label;
    std::vector<int> offsets;
};

struct TriggerPadData
{
    juce::String name;
    int midiNote = 36;
    bool muted = false;
    bool hasCustomSample = false;
    juce::String customFileName;
    juce::AudioBuffer<float> sampleBuffer;
    double sampleRate = 44100.0;
    int playhead = -1;
    float flashAlpha = 0.0f;

    float padSensitivity = 0.82f;
    bool isVoiceTrained = false;
    int trainedHits = 0;
    float featZCR = 0.0f;
    float featHighRatio = 0.0f;
    float featSustain = 0.0f;
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

        const int defaultNotes[8] = { 36, 38, 42, 39, 44, 40, 46, 35 };
        for (int i = 0; i < 8; ++i)
        {
            pads[i].name = "Pad " + juce::String (i + 1);
            pads[i].midiNote = defaultNotes[i];
            pads[i].hasCustomSample = false;
            pads[i].isVoiceTrained = false;
            pads[i].padSensitivity = 0.84f;
        }

        selectedDegreeVoicing.fill (0);
        applyMaqamPreset (0, 2);
        autoConnectDirectMidiDevice();
    }

    ~VoxN2AudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int) override
    {
        currentSampleRate = (sampleRate > 0.0) ? sampleRate : 44100.0;
        yinBuffer.assign (kYinWindowSize, 0.0f);
        yinWritePos = 0;
        synthPhase = 0.0;
        lpFilterState = 0.0f;
        hpCondenserState = 0.0f;
        prevSampleRaw = 0.0f;
        prevSamplePreEmp = 0.0f;
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void toggleAutoKeyDetection()
    {
        if (isDetectingKey.load())
        {
            finalizeAutoKeyDetection();
        }
        else
        {
            const juce::SpinLock::ScopedLockType sl (autoKeyLock);
            pitchClassWeights.fill (0.0f);
            quarterToneWeights.fill (0.0f);
            lastDetectedPitchClass = -1;
            totalDetectedVocalFrames = 0;
            autoKeySamplesElapsed = 0;
            autoKeyStatusText = "LISTENING...";
            isDetectingKey.store (true);
        }
    }

    void finalizeAutoKeyDetection()
    {
        isDetectingKey.store (false);
        const juce::SpinLock::ScopedLockType sl (autoKeyLock);

        if (totalDetectedVocalFrames < 8)
        {
            autoKeyStatusText = "TOO SHORT";
            return;
        }

        if (lastDetectedPitchClass >= 0 && lastDetectedPitchClass < 12)
            pitchClassWeights[lastDetectedPitchClass] *= 1.38f;

        int bestRoot = 0;
        int bestMaqam = 0;
        float bestScore = -1.0e9f;

        for (int r = 0; r < 12; ++r)
        {
            for (int m = 0; m < (int) kMaqamat.size(); ++m)
            {
                float score = pitchClassWeights[r] * 1.45f;
                for (int step = 0; step < 12; ++step)
                {
                    int pc = (r + step) % 12;
                    int rule = kMaqamat[m].intervals[step];
                    float w = pitchClassWeights[pc];
                    float qtW = quarterToneWeights[pc];

                    if (rule == 1) score += w * 1.0f - qtW * 0.5f;
                    else if (rule == 2) score += w * 0.85f + qtW * 1.85f;
                    else score -= w * 1.35f;
                }
                if (m == 9 || m == 8) score -= 0.05f;
                if (score > bestScore) { bestScore = score; bestRoot = r; bestMaqam = m; }
            }
        }

        applyMaqamPreset (bestRoot, bestMaqam);
        autoKeyStatusText = juce::String (kNoteNames[bestRoot]) + " " + juce::String (kMaqamat[bestMaqam].nameEn);
    }

    void autoConnectDirectMidiDevice()
    {
        auto devices = juce::MidiOutput::getAvailableDevices();
        for (int i = 0; i < devices.size(); ++i)
        {
            juce::String dName = devices[i].name;
            if ((dName.containsIgnoreCase ("loop") || dName.containsIgnoreCase ("voxn") || dName.containsIgnoreCase ("midi"))
                && ! dName.containsIgnoreCase ("Microsoft GS"))
            {
                openDirectMidiDeviceByIndex (i);
                return;
            }
        }
    }

    void cycleDirectMidiDevice()
    {
        auto devices = juce::MidiOutput::getAvailableDevices();
        if (devices.isEmpty())
        {
            const juce::SpinLock::ScopedLockType sl (directMidiLock);
            directMidiOut.reset();
            directMidiDeviceName = "No loopMIDI";
            return;
        }

        directMidiDeviceIndex = (directMidiDeviceIndex + 1) % (devices.size() + 1);
        if (directMidiDeviceIndex == devices.size())
        {
            const juce::SpinLock::ScopedLockType sl (directMidiLock);
            directMidiOut.reset();
            directMidiDeviceName = "Direct MIDI: OFF";
        }
        else
        {
            openDirectMidiDeviceByIndex (directMidiDeviceIndex);
        }
    }

    void openDirectMidiDeviceByIndex (int idx)
    {
        auto devices = juce::MidiOutput::getAvailableDevices();
        if (idx >= 0 && idx < devices.size())
        {
            auto opened = juce::MidiOutput::openDevice (devices[idx].identifier);
            const juce::SpinLock::ScopedLockType sl (directMidiLock);
            if (opened != nullptr)
            {
                directMidiOut = std::move (opened);
                directMidiDeviceName = devices[idx].name;
                directMidiDeviceIndex = idx;
            }
        }
    }

    void emitMidiEvent (const juce::MidiMessage& msg, juce::MidiBuffer& midiMessages, int sampleOffset = 0)
    {
        midiMessages.addEvent (msg, sampleOffset);
        {
            const juce::SpinLock::ScopedLockType sl (directMidiLock);
            if (directMidiOut != nullptr)
                directMidiOut->sendMessageNow (msg);
        }
        recordMidiEventInternal (msg);
    }

    void setExclusiveEngineMode (int newMode)
    {
        int clamped = juce::jlimit (0, 2, newMode);
        if (exclusiveMode.exchange (clamped) != clamped)
            pendingAllNotesOff.store (true);
    }

    juce::uint8 computeDynamicVelocity (float inputLevel, float floorLevel) const
    {
        if (fixedVelocity127) return 127;
        float norm = juce::jlimit (0.0f, 1.0f, (inputLevel - floorLevel) / juce::jmax (0.01f, 0.42f - floorLevel));
        float exponent = std::pow (2.2f, -velocityCurve);
        float shaped = std::pow (norm, exponent);
        int vMin = juce::jmin (minVelocity, maxVelocity);
        int vMax = juce::jmax (minVelocity, maxVelocity);
        int vel = vMin + (int) std::round (shaped * (float) (vMax - vMin));
        return (juce::uint8) juce::jlimit (1, 127, vel);
    }

    void startRoomCalibration()
    {
        calibPeakRms = 0.0f;
        calibSamplesRemaining = (int) (currentSampleRate * 1.0);
        isCalibratingRoom.store (true);
    }

    void startVoiceTrainingForPad (int padIndex)
    {
        if (padIndex < 0 || padIndex >= 8) return;
        setExclusiveEngineMode (ModeTriggersOnly);
        pads[padIndex].isVoiceTrained = false;
        pads[padIndex].trainedHits = 0;
        pads[padIndex].featZCR = 0.0f;
        pads[padIndex].featHighRatio = 0.0f;
        pads[padIndex].featSustain = 0.0f;
        trainingPadIndex.store (padIndex);
    }

    void clearVoiceTrainingForPad (int padIndex)
    {
        if (padIndex < 0 || padIndex >= 8) return;
        if (trainingPadIndex.load() == padIndex) trainingPadIndex.store (-1);
        pads[padIndex].isVoiceTrained = false;
        pads[padIndex].trainedHits = 0;
    }

    void startMidiCapture()
    {
        const juce::SpinLock::ScopedLockType sl (captureLock);
        capturedMidiSequence.clear();
        capturedSamplesElapsed = 0;
        captureFirstEventReceived = false;
        capturedNoteCount.store (0);
        isCapturingMidi.store (true);
    }

    void stopMidiCapture()
    {
        isCapturingMidi.store (false);
        const juce::SpinLock::ScopedLockType sl (captureLock);
        double bpm = hostBpm.load();
        double endTicks = ((double) capturedSamplesElapsed / currentSampleRate) * (bpm / 60.0) * 960.0;
        for (int ch = 1; ch <= 16; ++ch)
            for (int n = 0; n < 128; ++n)
                capturedMidiSequence.addEvent (juce::MidiMessage::noteOff (ch, n), endTicks);
        capturedMidiSequence.updateMatchedPairs();
    }

    void clearMidiCapture()
    {
        isCapturingMidi.store (false);
        const juce::SpinLock::ScopedLockType sl (captureLock);
        capturedMidiSequence.clear();
        capturedSamplesElapsed = 0;
        captureFirstEventReceived = false;
        capturedNoteCount.store (0);
    }

    void recordMidiEventInternal (const juce::MidiMessage& msg)
    {
        if (! isCapturingMidi.load()) return;
        const juce::SpinLock::ScopedLockType sl (captureLock);
        if (! captureFirstEventReceived)
        {
            captureFirstEventReceived = true;
            capturedSamplesElapsed = 0;
        }

        double bpm = hostBpm.load();
        double rawSec = (double) capturedSamplesElapsed / currentSampleRate;
        double compensatedSec = juce::jmax (0.0, rawSec - 0.015);
        double ticks = compensatedSec * (bpm / 60.0) * 960.0;

        if (quantizeMode == 1) ticks = std::round (ticks / 480.0) * 480.0;
        else if (quantizeMode == 2) ticks = std::round (ticks / 240.0) * 240.0;

        juce::MidiMessage copy (msg);
        copy.setTimeStamp (ticks);
        capturedMidiSequence.addEvent (copy);
        capturedMidiSequence.updateMatchedPairs();
        if (msg.isNoteOn()) capturedNoteCount.fetch_add (1);
    }

    juce::File exportCapturedMidiToFile (const juce::File& targetFile)
    {
        juce::MidiMessageSequence seqCopy;
        {
            const juce::SpinLock::ScopedLockType sl (captureLock);
            seqCopy = capturedMidiSequence;
        }
        if (seqCopy.getNumEvents() == 0)
        {
            seqCopy.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0.0);
            seqCopy.addEvent (juce::MidiMessage::noteOff (1, 60), 960.0);
        }
        seqCopy.updateMatchedPairs();
        juce::MidiFile midiFile;
        midiFile.setTicksPerQuarterNote (960);
        midiFile.addTrack (seqCopy);
        targetFile.deleteFile();
        juce::FileOutputStream out (targetFile);
        if (out.openedOk()) { midiFile.writeTo (out); out.flush(); }
        return targetFile;
    }

    juce::File exportCapturedMidiToTempFile()
    {
        return exportCapturedMidiToFile (juce::File::getSpecialLocation (juce::File::tempDirectory)
                                             .getChildFile ("VOXN2_CyberGold_Captured.mid"));
    }

    bool loadCustomSampleForPad (int padIndex, const juce::File& file)
    {
        if (padIndex < 0 || padIndex >= 8 || ! file.existsAsFile()) return false;
        std::unique_ptr<juce::AudioFormatReader> reader (formatManager.createReaderFor (file));
        if (reader == nullptr) return false;

        const int numSamples = (int) juce::jmin<juce::int64> (reader->lengthInSamples, (juce::int64) (reader->sampleRate * 8.0));
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

    void clearPadSample (int padIndex)
    {
        if (padIndex < 0 || padIndex >= 8) return;
        const juce::ScopedLock sl (sampleLock);
        pads[padIndex].hasCustomSample = false;
        pads[padIndex].customFileName.clear();
        pads[padIndex].sampleBuffer.setSize (0, 0);
        pads[padIndex].playhead = -1;
    }

    void triggerPadNow (int padIndex)
    {
        if (padIndex >= 0 && padIndex < 8) pendingPadTrigger.store (padIndex);
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
        rebuildSmartDegreeChords();
    }

    void rebuildSmartDegreeChords()
    {
        int deg = 0;
        for (int step = 0; step < 12 && deg < 7; ++step)
        {
            int pc = (rootNote + step) % 12;
            if (noteStates[pc] == 0) continue;

            degreePitchClasses[deg] = pc;
            auto& opts = degreeChordOptions[deg];
            opts.clear();

            juce::String nName = kNoteNames[pc];
            bool isRootQt = (noteStates[pc] == 2);

            auto hasInterval = [&] (int semi) -> bool { return noteStates[(pc + semi) % 12] == 1; };
            auto isIntervalQt = [&] (int semi) -> bool { return noteStates[(pc + semi) % 12] == 2; };

            if (isRootQt || isIntervalQt (4) || isIntervalQt (3) || isIntervalQt (7))
            {
                opts.push_back ({ "5th Power",     { 0, 7, 12 } });
                opts.push_back ({ "Sus4 Oriental", { 0, 5, 7 } });
                opts.push_back ({ "Sus2 Oriental", { 0, 2, 7 } });
                opts.push_back ({ "Octave Double", { 0, 12 } });
                int rootDiff = (rootNote - pc - 12);
                opts.push_back ({ "Pedal Root",    { rootDiff, 0, 12 } });
            }
            else
            {
                bool hasMaj3 = hasInterval (4), hasMin3 = hasInterval (3);
                bool hasP5   = hasInterval (7), hasDim5 = hasInterval (6);
                bool hasMaj7 = hasInterval (11), hasMin7 = hasInterval (10);
                bool hasMaj2 = hasInterval (2), hasP4   = hasInterval (5), hasMaj6 = hasInterval (9);

                if (hasMaj3 && hasP5)
                {
                    opts.push_back ({ nName + " Maj",   { 0, 4, 7 } });
                    if (hasMaj7) opts.push_back ({ nName + " Maj7",  { 0, 4, 7, 11 } });
                    if (hasMin7) opts.push_back ({ nName + " Dom7",  { 0, 4, 7, 10 } });
                    if (hasMaj2) opts.push_back ({ nName + " Add9",  { 0, 4, 7, 14 } });
                    if (hasMaj6) opts.push_back ({ nName + " 6th",   { 0, 4, 7, 9 } });
                }
                else if (hasMin3 && hasP5)
                {
                    opts.push_back ({ nName + " Min",   { 0, 3, 7 } });
                    if (hasMin7) opts.push_back ({ nName + " Min7",  { 0, 3, 7, 10 } });
                    if (hasMaj7) opts.push_back ({ nName + " mMaj7", { 0, 3, 7, 11 } });
                    if (hasMaj2) opts.push_back ({ nName + " Min9",  { 0, 3, 7, 10, 14 } });
                    if (hasMaj6) opts.push_back ({ nName + " Min6",  { 0, 3, 7, 9 } });
                }
                else if (hasMin3 && hasDim5)
                {
                    opts.push_back ({ nName + " Dim",   { 0, 3, 6 } });
                    if (hasMin7) opts.push_back ({ nName + " m7b5",  { 0, 3, 6, 10 } });
                    opts.push_back ({ "Dom7 Inv",      { 0, 3, 5, 9 } });
                }

                if (hasMaj2 && hasP5) opts.push_back ({ nName + " Sus2",  { 0, 2, 7 } });
                if (hasP4   && hasP5) opts.push_back ({ nName + " Sus4",  { 0, 5, 7 } });
                if (hasP5)            opts.push_back ({ nName + " Power", { 0, 7, 12 } });
            }

            if (opts.empty()) opts.push_back ({ nName + " Oct", { 0, 12 } });

            selectedDegreeVoicing[deg] = juce::jlimit (0, (int) opts.size() - 1, selectedDegreeVoicing[deg]);
            ++deg;
        }
    }

    float detectPitchYIN (const std::vector<float>& buf, double sr, float& outConfidence)
    {
        const int W = (int) buf.size() / 2;
        std::vector<float> diff (W, 0.0f), cmnd (W, 0.0f);
        for (int tau = 1; tau < W; ++tau)
        {
            float sum = 0.0f;
            for (int i = 0; i < W; ++i) { float d = buf[i] - buf[i + tau]; sum += d * d; }
            diff[tau] = sum;
        }
        cmnd[0] = 1.0f;
        float runningSum = 0.0f;
        for (int tau = 1; tau < W; ++tau)
        {
            runningSum += diff[tau];
            cmnd[tau] = (runningSum > 1.0e-9f) ? (diff[tau] * (float) tau / runningSum) : 1.0f;
        }
        const int minTau = juce::jmax (2, (int) (sr / 1050.0));
        const int maxTau = juce::jmin (W - 2, (int) (sr / 65.0));
        int bestTau = -1;
        float threshold = condenserMode ? 0.12f : 0.16f;

        for (int tau = minTau; tau < maxTau; ++tau)
        {
            if (cmnd[tau] < threshold)
            {
                while (tau + 1 < maxTau && cmnd[tau + 1] < cmnd[tau]) ++tau;
                bestTau = tau;
                break;
            }
        }
        if (bestTau < 0) { outConfidence = 0.0f; return -1.0f; }

        outConfidence = 1.0f - cmnd[bestTau];
        float s0 = cmnd[bestTau - 1], s1 = cmnd[bestTau], s2 = cmnd[bestTau + 1];
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

        if (auto* ph = getPlayHead())
        {
            if (auto pos = ph->getPosition())
            {
                if (auto bpmOpt = pos->getBpm())
                    if (*bpmOpt > 20.0 && *bpmOpt < 350.0)
                        hostBpm.store (*bpmOpt);
            }
        }

        midiMessages.clear();
        if (isCapturingMidi.load() && captureFirstEventReceived)
            capturedSamplesElapsed += numSamples;

        std::vector<float> micMono (numSamples, 0.0f);
        if (numInChannels > 0)
        {
            const float* inPtr = buffer.getReadPointer (0);
            for (int i = 0; i < numSamples; ++i)
            {
                float raw = inPtr[i];
                if (condenserMode)
                {
                    hpCondenserState = 0.988f * (hpCondenserState + raw - prevSampleRaw);
                    prevSampleRaw = raw;
                    micMono[i] = hpCondenserState;
                }
                else
                {
                    micMono[i] = raw;
                }
            }
        }

        for (int ch = 0; ch < numOutChannels; ++ch) buffer.clear (ch, 0, numSamples);

        if (pendingAllNotesOff.exchange (false)) stopActiveMelodyAndChords (midiMessages);

        float sumSq = 0.0f, highEnergy = 0.0f, lowEnergy = 0.0f, peakAbs = 0.0f, preEmpSumSq = 0.0f;
        int zeroCrossings = 0;
        int firstTransientOffset = 0;
        bool foundTransientSample = false;

        float gateOpenThresh  = 0.002f + (globalGateLevel * globalGateLevel) * 0.22f;
        float gateCloseThresh = gateOpenThresh * 0.72f;

        for (int i = 0; i < numSamples; ++i)
        {
            float s = micMono[i];
            sumSq += s * s;
            float absS = std::abs (s);
            peakAbs = juce::jmax (peakAbs, absS);

            if (! foundTransientSample && absS > gateOpenThresh * 1.4f)
            {
                firstTransientOffset = i;
                foundTransientSample = true;
            }

            float preEmp = s - 0.94f * prevSamplePreEmp;
            prevSamplePreEmp = s;
            preEmpSumSq += preEmp * preEmp;

            lpFilterState = 0.85f * lpFilterState + 0.15f * s;
            float hp = s - lpFilterState;
            lowEnergy  += lpFilterState * lpFilterState;
            highEnergy += hp * hp;

            if (i > 0 && ((micMono[i - 1] >= 0.0f && s < 0.0f) || (micMono[i - 1] < 0.0f && s >= 0.0f)))
                ++zeroCrossings;

            yinBuffer[yinWritePos] = s;
            yinWritePos = (yinWritePos + 1) % kYinWindowSize;
        }

        float nSamp = (float) juce::jmax (1, numSamples);
        float rms = std::sqrt (sumSq / nSamp);
        float hfRms = std::sqrt (preEmpSumSq / nSamp);

        if (isCalibratingRoom.load())
        {
            calibPeakRms = juce::jmax (calibPeakRms, rms);
            calibSamplesRemaining -= numSamples;
            if (calibSamplesRemaining <= 0)
            {
                float targetGate = calibPeakRms * 1.45f + 0.004f;
                globalGateLevel = juce::jlimit (0.04f, 0.85f, std::sqrt (juce::jmax (0.0f, (targetGate - 0.002f) / 0.22f)));
                isCalibratingRoom.store (false);
            }
        }

        if (! globalGateIsOpen && rms > gateOpenThresh) globalGateIsOpen = true;
        else if (globalGateIsOpen && rms < gateCloseThresh) globalGateIsOpen = false;

        float hiHatMultiplier = 1.5f + hiHatBoost * 4.5f;
        float effectiveTrigLevel = juce::jmax (rms, hfRms * hiHatMultiplier);
        currentInputRms.store (rms);

        float zcr = (float) zeroCrossings / nSamp;
        float highRatio = highEnergy / (lowEnergy + highEnergy + 1.0e-7f);
        float sustainRatio = (peakAbs > 1.0e-5f) ? (rms / peakAbs) : 0.0f;
        vowelEnv = 0.82f * vowelEnv + 0.18f * juce::jlimit (0.0f, 1.0f, rms * 8.5f);

        const int mode = exclusiveMode.load();
        int manualPad = pendingPadTrigger.exchange (-1);
        if (manualPad >= 0 && manualPad < 8) fireTriggerPad (manualPad, 115, midiMessages, 0);

        if (mode == ModeTriggersOnly)
        {
            if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty()) stopActiveMelodyAndChords (midiMessages);

            if (triggerCooldownSamples > 0) triggerCooldownSamples -= numSamples;

            float onsetDelta = effectiveTrigLevel - prevTrigLevel;
            prevTrigLevel = effectiveTrigLevel;

            int trainIdx = trainingPadIndex.load();
            bool isTrainingNow = (trainIdx >= 0 && trainIdx < 8);
            float activeSens = isTrainingNow ? trainSensitivity : triggerSensitivity;

            float inv = 1.0f - juce::jlimit (0.0f, 0.99f, activeSens);
            float baseThresh = 0.0015f + (inv * inv) * 0.11f;
            float minGateFloor = juce::jmax (gateOpenThresh * 0.65f, 0.0008f + (inv * inv) * 0.025f);

            if (onsetDelta > baseThresh && effectiveTrigLevel > minGateFloor && triggerCooldownSamples <= 0)
            {
                juce::uint8 dynVel = computeDynamicVelocity (effectiveTrigLevel, minGateFloor);

                if (isTrainingNow)
                {
                    auto& tp = pads[trainIdx];
                    float n = (float) tp.trainedHits;
                    tp.featZCR       = (tp.featZCR * n + zcr) / (n + 1.0f);
                    tp.featHighRatio = (tp.featHighRatio * n + highRatio) / (n + 1.0f);
                    tp.featSustain   = (tp.featSustain * n + sustainRatio) / (n + 1.0f);
                    tp.trainedHits++;
                    tp.isVoiceTrained = true;
                    fireTriggerPad (trainIdx, dynVel, midiMessages, firstTransientOffset);

                    if (tp.trainedHits >= 3) trainingPadIndex.store (-1);

                    triggerCooldownSamples = (int) (currentSampleRate * 0.13);
                }
                else
                {
                    int bestPad = -1;
                    float bestScore = 999.0f;

                    for (int p = 0; p < 8; ++p)
                    {
                        if (pads[p].isVoiceTrained)
                        {
                            float dz = (zcr - pads[p].featZCR) * 3.0f;
                            float dh = (highRatio - pads[p].featHighRatio) * 2.8f;
                            float ds = (sustainRatio - pads[p].featSustain) * 1.2f;
                            float rawDist = std::sqrt (dz * dz + dh * dh + ds * ds);
                            float padBoostFactor = 1.35f - (pads[p].padSensitivity * 0.75f);
                            float weightedDist = rawDist * padBoostFactor;

                            if (weightedDist < bestScore) { bestScore = weightedDist; bestPad = p; }
                        }
                    }

                    float maxAllowedDistance = 0.25f + trainSensitivity * 1.65f;
                    if (bestPad >= 0 && bestScore <= maxAllowedDistance)
                    {
                        fireTriggerPad (bestPad, dynVel, midiMessages, firstTransientOffset);
                        triggerCooldownSamples = (int) (currentSampleRate * 0.075);
                    }
                    else if (bestPad < 0)
                    {
                        int fallbackPad = (highRatio > 0.65f || zcr > 0.20f) ? 2 : ((highRatio > 0.28f) ? 1 : 0);
                        fireTriggerPad (fallbackPad, dynVel, midiMessages, firstTransientOffset);
                        triggerCooldownSamples = (int) (currentSampleRate * 0.075);
                    }
                }
            }
        }
        else
        {
            prevTrigLevel = effectiveTrigLevel;

            int manualNote = guiManualMidiNote.load();
            float detectedFreq = -1.0f;
            float pitchConfidence = 1.0f;

            if (manualNote >= 0)
            {
                detectedFreq = 440.0f * std::pow (2.0f, (manualNote - 69.0f) / 12.0f);
            }
            else if (globalGateIsOpen)
            {
                std::vector<float> orderedYin (kYinWindowSize);
                for (int i = 0; i < kYinWindowSize; ++i)
                    orderedYin[i] = yinBuffer[(yinWritePos + i) % kYinWindowSize];
                detectedFreq = detectPitchYIN (orderedYin, currentSampleRate, pitchConfidence);
            }

            bool validVocalPitch = (detectedFreq > 60.0f && detectedFreq < 1050.0f)
                                   && (! condenserMode || pitchConfidence > 0.72f || manualNote >= 0);

            if (isDetectingKey.load())
            {
                bool shouldFinishNow = false;
                {
                    const juce::SpinLock::ScopedLockType sl (autoKeyLock);
                    autoKeySamplesElapsed += numSamples;
                    if (validVocalPitch && manualNote < 0)
                    {
                        float rawExactMidi = 69.0f + 12.0f * std::log2 (detectedFreq / 440.0f);
                        int nearestNote = (int) std::round (rawExactMidi);
                        float devCents = (rawExactMidi - (float) nearestNote) * 100.0f;
                        int pc = ((nearestNote % 12) + 12) % 12;

                        if (devCents < -24.0f && devCents > -68.0f) quarterToneWeights[pc] += 1.0f;

                        pitchClassWeights[pc] += 1.0f;
                        lastDetectedPitchClass = pc;
                        totalDetectedVocalFrames++;
                    }
                    if (autoKeySamplesElapsed >= (int) (currentSampleRate * 4.5) && totalDetectedVocalFrames >= 12)
                        shouldFinishNow = true;
                }
                if (shouldFinishNow) finalizeAutoKeyDetection();
            }

            if (validVocalPitch)
            {
                float rawMidi = 69.0f + 12.0f * std::log2 (detectedFreq / 440.0f);
                rawMidi += (float) semitoneTranspose;

                int candidateMidi = (int) std::round (rawMidi);

                if (vocalRangePreset > RangeFollow && manualNote < 0)
                {
                    int minMidi = 48, maxMidi = 72;
                    switch (vocalRangePreset)
                    {
                        case RangeBass:     minMidi = 24; maxMidi = 47; break;
                        case RangeBaritone: minMidi = 45; maxMidi = 68; break;
                        case RangeTenor:    minMidi = 48; maxMidi = 71; break;
                        case RangeAlto:     minMidi = 53; maxMidi = 76; break;
                        case RangeSoprano:  minMidi = 60; maxMidi = 84; break;
                        default: break;
                    }
                    while (candidateMidi < minMidi) candidateMidi += 12;
                    while (candidateMidi > maxMidi) candidateMidi -= 12;
                }

                candidateMidi = juce::jlimit (12, 115, candidateMidi);

                float hysteresisWidth = 0.45f + (stickiness * 0.48f);
                if (activeRawMidiNote >= 0 && std::abs ((float) candidateMidi - (float) activeRawMidiNote) < hysteresisWidth)
                    candidateMidi = activeRawMidiNote;

                activeRawMidiNote = candidateMidi;
                int snappedMidi = snapToMaqamScale (activeRawMidiNote);
                int pc = ((snappedMidi % 12) + 12) % 12;

                float targetCents = (noteStates[pc] == 2) ? quarterToneCents : 0.0f;
                float maxBendCents = (float) pbRangeSemitones * 100.0f;
                int pbValue = 8192 + (int) std::round ((targetCents / maxBendCents) * 8192.0f);
                pbValue = juce::jlimit (0, 16383, pbValue);
                currentPitchBendValue.store (pbValue);
                currentCentsOffset.store ((int) std::round (targetCents));

                if (snappedMidi != activeOutputMidiNote)
                {
                    stopActiveMelodyAndChords (midiMessages);
                    juce::uint8 dynVel = (manualNote >= 0) ? 110 : computeDynamicVelocity (rms, gateOpenThresh);
                    activeOutputMidiNote = snappedMidi;
                    currentDetectedMidiNote.store (snappedMidi);

                    if (mode == ModeMelodyOnly)
                    {
                        emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), midiMessages, 0);
                        emitMidiEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, dynVel), midiMessages, 0);
                    }
                    else if (mode == ModeChordsOnly)
                    {
                        triggerSmartOrientalChord (snappedMidi, dynVel, midiMessages);
                    }
                }
                else if (mode == ModeMelodyOnly)
                {
                    emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), midiMessages, 0);
                }
            }
            else
            {
                if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty()) stopActiveMelodyAndChords (midiMessages);
            }
        }

        float* outL = buffer.getWritePointer (0);
        float* outR = (numOutChannels > 1) ? buffer.getWritePointer (1) : outL;

        if (! intellAudioMuted.load() && activeOutputMidiNote >= 0 && mode != ModeTriggersOnly)
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
                if (pad.playhead >= 0 && ! pad.muted && pad.hasCustomSample && pad.sampleBuffer.getNumSamples() > 0)
                {
                    const float* samp = pad.sampleBuffer.getReadPointer (0);
                    const int totalLen = pad.sampleBuffer.getNumSamples();
                    for (int i = 0; i < numSamples; ++i)
                    {
                        if (pad.playhead < totalLen)
                        {
                            float v = samp[pad.playhead++] * lastPadGain[p];
                            outL[i] += v;
                            outR[i] += v;
                        }
                        else { pad.playhead = -1; break; }
                    }
                }
            }
        }
    }

    void stopActiveMelodyAndChords (juce::MidiBuffer& midiMessages)
    {
        if (activeOutputMidiNote >= 0)
        {
            emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, activeOutputMidiNote), midiMessages, 0);
            emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, 8192), midiMessages, 0);
        }
        stopActiveChord (midiMessages);
        activeOutputMidiNote = -1;
        activeRawMidiNote = -1;
        currentDetectedMidiNote.store (-1);
        currentPitchBendValue.store (8192);
        currentCentsOffset.store (0);
    }

    int snapToMaqamScale (int midiNote) const
    {
        int pc = ((midiNote % 12) + 12) % 12;
        if (noteStates[pc] > 0) return midiNote;
        for (int dist = 1; dist <= 6; ++dist)
        {
            if (noteStates[(pc - dist + 12) % 12] > 0) return midiNote - dist;
            if (noteStates[(pc + dist) % 12] > 0)      return midiNote + dist;
        }
        return midiNote;
    }

    void fireTriggerPad (int padIdx, juce::uint8 velocity, juce::MidiBuffer& midiMessages, int sampleOffset = 0)
    {
        auto& pad = pads[padIdx];
        pad.flashAlpha = 1.0f;
        lastPadGain[padIdx] = (float) velocity / 127.0f * 0.75f;
        if (pad.hasCustomSample) pad.playhead = 0;

        emitMidiEvent (juce::MidiMessage::noteOn  (triggersMidiChannel, pad.midiNote, velocity), midiMessages, sampleOffset);
        emitMidiEvent (juce::MidiMessage::noteOff (triggersMidiChannel, pad.midiNote), midiMessages, juce::jmin (sampleOffset + 64, 127));
    }

    void triggerSmartOrientalChord (int rootMidi, juce::uint8 vel, juce::MidiBuffer& midiMessages)
    {
        stopActiveChord (midiMessages);
        int pc = ((rootMidi % 12) + 12) % 12;

        int foundDeg = 0;
        for (int d = 0; d < 7; ++d)
        {
            if (degreePitchClasses[d] == pc) { foundDeg = d; break; }
        }

        const auto& opts = degreeChordOptions[foundDeg];
        int choiceIdx = juce::jlimit (0, juce::jmax (0, (int) opts.size() - 1), selectedDegreeVoicing[foundDeg]);

        if (! opts.empty())
        {
            for (int offset : opts[choiceIdx].offsets)
                activeChordNotes.push_back (juce::jlimit (12, 115, rootMidi + offset));
        }
        else
        {
            activeChordNotes = { rootMidi, rootMidi + 7, rootMidi + 12 };
        }

        for (int n : activeChordNotes)
            emitMidiEvent (juce::MidiMessage::noteOn (chordsMidiChannel, n, (juce::uint8) (vel * 0.85f)), midiMessages, 0);
    }

    void auditionDegreeChord (int degIndex)
    {
        if (degIndex < 0 || degIndex >= 7) return;
        int baseMidi = 60 + degreePitchClasses[degIndex];
        if (baseMidi < 60 + rootNote) baseMidi += 12;
        setExclusiveEngineMode (ModeChordsOnly);
        guiManualMidiNote.store (baseMidi);
    }

    void stopActiveChord (juce::MidiBuffer& midiMessages)
    {
        for (int n : activeChordNotes)
            emitMidiEvent (juce::MidiMessage::noteOff (chordsMidiChannel, n), midiMessages, 0);
        activeChordNotes.clear();
    }

    float renderInternalSynthSample (double phase) const
    {
        double twoPi = juce::MathConstants<double>::twoPi;
        switch (synthPresetIndex)
        {
            case 0: return (float) (std::sin (twoPi * phase) * 0.55 + std::sin (twoPi * phase * 2.0) * 0.30 + std::sin (twoPi * phase * 3.0) * 0.15);
            case 1: return (float) (std::sin (twoPi * phase) * 0.65 + std::sin (twoPi * phase * 3.0) * 0.25);
            case 2: return (float) (std::sin (twoPi * phase) * 0.75 + std::sin (twoPi * phase * 2.0) * 0.20);
            default: return (float) std::sin (twoPi * phase);
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
        state.setProperty ("vocalRangePreset", vocalRangePreset, nullptr);
        state.setProperty ("semitoneTranspose", semitoneTranspose, nullptr);
        state.setProperty ("velocityCurve", velocityCurve, nullptr);
        state.setProperty ("fixedVelocity127", fixedVelocity127, nullptr);
        state.setProperty ("globalGateLevel", globalGateLevel, nullptr);
        state.setProperty ("condenserMode", condenserMode, nullptr);
        state.setProperty ("quantizeMode", quantizeMode, nullptr);
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
            currentMaqamIndex = state.getProperty ("maqamIndex", 2);
            quarterToneCents = state.getProperty ("quarterToneCents", -50.0f);
            vocalRangePreset = state.getProperty ("vocalRangePreset", 0);
            semitoneTranspose = state.getProperty ("semitoneTranspose", 0);
            velocityCurve = state.getProperty ("velocityCurve", 0.35f);
            fixedVelocity127 = state.getProperty ("fixedVelocity127", false);
            globalGateLevel = state.getProperty ("globalGateLevel", 0.16f);
            condenserMode = state.getProperty ("condenserMode", true);
            quantizeMode = state.getProperty ("quantizeMode", 2);
            applyMaqamPreset (rootNote, currentMaqamIndex);
        }
    }

    std::array<TriggerPadData, 8> pads;
    std::array<int, 12> noteStates { 1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2 };
    std::array<int, 7> degreePitchClasses { 0, 2, 4, 5, 7, 9, 11 };
    std::array<std::vector<ChordVoicingOption>, 7> degreeChordOptions;
    std::array<int, 7> selectedDegreeVoicing { 0, 0, 0, 0, 0, 0, 0 };

    int rootNote = 0;
    int currentMaqamIndex = 2;
    float quarterToneCents = -50.0f;
    float stickiness = 0.65f;

    float velocityCurve = 0.35f;
    int minVelocity = 35;
    int maxVelocity = 127;
    bool fixedVelocity127 = false;

    std::atomic<double> hostBpm { 120.0 };
    int quantizeMode = 2;

    int vocalRangePreset = RangeFollow;
    int semitoneTranspose = 0;

    float globalGateLevel = 0.16f;
    bool condenserMode = true;
    std::atomic<bool> isCalibratingRoom { false };

    std::atomic<bool> isDetectingKey { false };
    juce::String autoKeyStatusText { "AUTO KEY" };

    float triggerSensitivity = 0.86f;
    float trainSensitivity   = 0.90f;
    float hiHatBoost         = 0.85f;
    int pbRangeSemitones = 2;
    int synthPresetIndex = 0;
    int pitchMidiChannel = 1, chordsMidiChannel = 1, triggersMidiChannel = 1;
    float vowelEnv = 0.50f;

    std::atomic<int>   exclusiveMode { ModeMelodyOnly };
    std::atomic<int>   trainingPadIndex { -1 };
    std::atomic<bool>  pendingAllNotesOff { false };
    std::atomic<bool>  isCapturingMidi { false };
    std::atomic<int>   capturedNoteCount { 0 };
    std::atomic<bool>  intellAudioMuted { false };
    std::atomic<bool>  trigAudioMuted   { false };
    std::atomic<int>   currentDetectedMidiNote { -1 };
    std::atomic<int>   currentPitchBendValue { 8192 };
    std::atomic<int>   currentCentsOffset { 0 };
    std::atomic<float> currentInputRms { 0.0f };
    std::atomic<int>   guiManualMidiNote { -1 };
    std::atomic<int>   pendingPadTrigger { -1 };

    juce::String directMidiDeviceName { "Click to Select loopMIDI" };

private:
    static constexpr int kYinWindowSize = 512;
    std::vector<float> yinBuffer;
    int yinWritePos = 0;
    double currentSampleRate = 44100.0;
    double synthPhase = 0.0;
    float lpFilterState = 0.0f;
    float hpCondenserState = 0.0f;
    float prevSampleRaw = 0.0f;
    float prevSamplePreEmp = 0.0f;
    bool globalGateIsOpen = false;

    float calibPeakRms = 0.0f;
    int calibSamplesRemaining = 0;

    juce::SpinLock autoKeyLock;
    std::array<float, 12> pitchClassWeights {};
    std::array<float, 12> quarterToneWeights {};
    int lastDetectedPitchClass = -1;
    int totalDetectedVocalFrames = 0;
    int autoKeySamplesElapsed = 0;

    int activeRawMidiNote = -1, activeOutputMidiNote = -1;
    float prevTrigLevel = 0.0f;
    int triggerCooldownSamples = 0;
    std::array<float, 8> lastPadGain { 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f };

    std::vector<int> activeChordNotes;
    juce::AudioFormatManager formatManager;
    juce::CriticalSection sampleLock;

    juce::SpinLock directMidiLock;
    std::unique_ptr<juce::MidiOutput> directMidiOut;
    int directMidiDeviceIndex = -1;

    juce::SpinLock captureLock;
    juce::MidiMessageSequence capturedMidiSequence;
    juce::int64 capturedSamplesElapsed = 0;
    bool captureFirstEventReceived = false;

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

    ~VoxN2AudioProcessorEditor() override { stopTimer(); }

    void timerCallback() override
    {
        for (auto& pad : proc.pads)
            if (pad.flashAlpha > 0.01f) pad.flashAlpha *= 0.82f;
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
            "Load Custom Audio (.wav, .mp3)", juce::File{}, "*.wav;*.mp3;*.flac;*.ogg");
        fileChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this, padIndex] (const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file.existsAsFile()) { proc.loadCustomSampleForPad (padIndex, file); repaint(); }
            });
    }

    void drawCyberGoldMonogramLogo (juce::Graphics& g, juce::Rectangle<float> area, juce::Colour gold, juce::Colour bright)
    {
        float x = area.getX(), y = area.getY(), w = area.getWidth(), h = area.getHeight();
        juce::Path monogram;
        monogram.startNewSubPath (x + w * 0.08f, y + h * 0.18f);
        monogram.lineTo        (x + w * 0.32f, y + h * 0.84f);
        monogram.lineTo        (x + w * 0.48f, y + h * 0.36f);
        monogram.lineTo        (x + w * 0.56f, y + h * 0.62f);
        monogram.lineTo        (x + w * 0.66f, y + h * 0.24f);
        monogram.lineTo        (x + w * 0.92f, y + h * 0.24f);
        monogram.moveTo        (x + w * 0.66f, y + h * 0.24f);
        monogram.lineTo        (x + w * 0.60f, y + h * 0.78f);
        monogram.lineTo        (x + w * 0.90f, y + h * 0.78f);
        monogram.lineTo        (x + w * 0.90f, y + h * 0.52f);
        monogram.lineTo        (x + w * 0.74f, y + h * 0.52f);

        g.setColour (gold.withAlpha (0.28f));
        g.strokePath (monogram, juce::PathStrokeType (4.2f, juce::PathStrokeType::mitered, juce::PathStrokeType::butt));
        g.setColour (bright);
        g.strokePath (monogram, juce::PathStrokeType (2.0f, juce::PathStrokeType::mitered, juce::PathStrokeType::butt));
    }

    void paint (juce::Graphics& g) override
    {
        const juce::Colour bgDark (0xff1b1b1f), cardDark (0xff25252b), cyberGold (0xffe5b84b);
        const juce::Colour neonPink (0xfff03e76), neonGreen (0xff2ecc71), textBright (0xfff4f0e8), textMuted (0xff8e8d96);

        g.fillAll (bgDark);
        g.setColour (juce::Colour (0xff141417));
        g.fillRect (0, 0, getWidth(), 42);

        drawCyberGoldMonogramLogo (g, { 10.0f, 7.0f, 32.0f, 28.0f }, cyberGold, cyberGold);
        g.setFont (juce::Font (14.5f, juce::Font::bold));
        g.setColour (textBright);
        g.drawText ("VOXN 2", 46, 8, 56, 26, juce::Justification::centredLeft);

        const int activeMode = proc.exclusiveMode.load();
        const char* modeNames[3] = { "1. MELODY ONLY", "2. TRIGGERS ONLY", "3. CHORDS ONLY" };
        for (int m = 0; m < 3; ++m)
        {
            exclusiveModeRects[m] = { 106.0f + m * 120.0f, 8.0f, 114.0f, 26.0f };
            bool isCurr = (activeMode == m);
            g.setColour (isCurr ? (m == 1 ? neonPink : cyberGold) : juce::Colour (0xff23232a));
            g.fillRoundedRectangle (exclusiveModeRects[m], 5.0f);
            g.setColour (isCurr ? juce::Colours::black : textBright);
            g.setFont (juce::Font (10.5f, juce::Font::bold));
            g.drawText (modeNames[m], exclusiveModeRects[m], juce::Justification::centred);
        }

        bool isRec = proc.isCapturingMidi.load();
        int noteCnt = proc.capturedNoteCount.load();

        midiRecButtonRect = { 474.0f, 8.0f, 102.0f, 26.0f };
        g.setColour (isRec ? neonPink : juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (midiRecButtonRect, 5.0f);
        g.setColour (isRec ? textBright : neonPink);
        g.fillEllipse (midiRecButtonRect.getX() + 10.0f, midiRecButtonRect.getY() + 9.0f, 8.0f, 8.0f);
        g.setColour (textBright);
        g.drawText (isRec ? ("STOP [" + juce::String (noteCnt) + "]") : "REC MIDI",
                    midiRecButtonRect.withTrimmedLeft (16.0f), juce::Justification::centred);

        quantizeCycleRect = { 582.0f, 8.0f, 106.0f, 26.0f };
        g.setColour (proc.quantizeMode > 0 ? juce::Colour (0xff2d2a1e) : juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (quantizeCycleRect, 5.0f);
        g.setColour (cyberGold);
        const char* qLabels[3] = { "Quant: OFF", "Quant: 1/8", "Quant: 1/16" };
        g.drawText (qLabels[proc.quantizeMode % 3], quantizeCycleRect, juce::Justification::centred);

        midiDragButtonRect = { 694.0f, 8.0f, 186.0f, 26.0f };
        g.setColour (noteCnt > 0 ? neonGreen : cyberGold);
        g.fillRoundedRectangle (midiDragButtonRect, 5.0f);
        g.setColour (juce::Colours::black);
        g.drawText ("DRAG MIDI TO DAW (" + juce::String (noteCnt
