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

struct MaqamPreset
{
    const char* nameEn;
    const char* nameAr;
    std::array<int, 12> intervals; // 0 = Off, 1 = Natural, 2 = Quarter-Tone Flat (1/2b)
};

// 10 سلالم ومقامات تشمل الميجور والماينور والمقامات الشرقية
static const std::array<MaqamPreset, 10> kMaqamat = {{
    { "Major",    "ميجور",  {1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1} },
    { "Minor",    "ماينور", {1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0} },
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
    int midiNote = 36;
    bool muted = false;
    bool hasCustomSample = false;
    juce::String customFileName;
    juce::AudioBuffer<float> sampleBuffer;
    double sampleRate = 44100.0;
    int playhead = -1;
    float flashAlpha = 0.0f;

    // بصمة الصوت لتدريب البيت بوكس (Beatbox Voice Fingerprint)
    bool isVoiceTrained = false;
    int trainedHits = 0;
    float featZCR = 0.0f;       // معدل تقاطع الصفر (يميز الدم عن التك والهاي هات)
    float featHighRatio = 0.0f; // نسبة الترددات العالية للمنخفضة
    float featSustain = 0.0f;   // طول النفس/الطنين (يميز البيز Bzzz عن الضربات الخاطفة)
};

// ==============================================================================
// المعالج الصوتي الرئيسي (VoxN2AudioProcessor)
// ==============================================================================
class VoxN2AudioProcessor : public juce::AudioProcessor
{
public:
    VoxN2AudioProcessor()
        : AudioProcessor (BusesProperties()
                          .withInput  ("Microphone Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Audio Output",     juce::AudioChannelSet::stereo(), true))
    {
        formatManager.registerBasicFormats();

        // مربعات فارغة تماماً مثل FPC بدون أي أصوات مدمجة
        const int defaultNotes[8] = { 36, 38, 42, 39, 44, 40, 46, 35 };
        for (int i = 0; i < 8; ++i)
        {
            pads[i].name = "Pad " + juce::String (i + 1);
            pads[i].midiNote = defaultNotes[i];
            pads[i].hasCustomSample = false;
            pads[i].isVoiceTrained = false;
        }

        applyMaqamPreset (0, 2); // Default: C Rast
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
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
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
            directMidiDeviceName = "No loopMIDI Found";
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

    // ==========================================================================
    // بدء أو مسح تدريب الصوت للبيت بوكس على الـ Pad المختار
    // ==========================================================================
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
        if (trainingPadIndex.load() == padIndex)
            trainingPadIndex.store (-1);
        pads[padIndex].isVoiceTrained = false;
        pads[padIndex].trainedHits = 0;
    }

    // ==========================================================================
    // تسجيل وسحب الـ MIDI المباشر (MIDI Capture)
    // ==========================================================================
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
        double endTicks = ((double) capturedSamplesElapsed / currentSampleRate) * 1920.0;
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
        double ticks = ((double) capturedSamplesElapsed / currentSampleRate) * 1920.0;
        juce::MidiMessage copy (msg);
        copy.setTimeStamp (ticks);
        capturedMidiSequence.addEvent (copy);
        capturedMidiSequence.updateMatchedPairs();
        if (msg.isNoteOn())
            capturedNoteCount.fetch_add (1);
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

    // تحميل أو مسح عينة WAV/MP3 على الـ Pad (نظام FPC)
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
        if (padIndex >= 0 && padIndex < 8)
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
        const int minTau = juce::jmax (2, (int) (sr / 1000.0));
        const int maxTau = juce::jmin (W - 2, (int) (sr / 65.0));
        int bestTau = -1;
        for (int tau = minTau; tau < maxTau; ++tau)
        {
            if (cmnd[tau] < 0.14f)
            {
                while (tau + 1 < maxTau && cmnd[tau + 1] < cmnd[tau]) ++tau;
                bestTau = tau;
                break;
            }
        }
        if (bestTau < 0) return -1.0f;
        float s0 = cmnd[bestTau - 1], s1 = cmnd[bestTau], s2 = cmnd[bestTau + 1];
        float denom = 2.0f * (2.0f * s1 - s2 - s0);
        float betterTau = (float) bestTau + ((std::abs (denom) > 1e-6f) ? (s2 - s0) / denom : 0.0f);
        return (betterTau > 0.0f) ? (float) (sr / betterTau) : -1.0f;
    }

    // ==========================================================================
    // معالجة الصوت وتحويل الأوكتاف وتدريب البيت بوكس (processBlock)
    // ==========================================================================
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int numSamples = buffer.getNumSamples();
        const int numInChannels = getTotalNumInputChannels();
        const int numOutChannels = getTotalNumOutputChannels();

        midiMessages.clear();
        if (isCapturingMidi.load() && captureFirstEventReceived)
            capturedSamplesElapsed += numSamples;

        std::vector<float> micMono (numSamples, 0.0f);
        if (numInChannels > 0)
        {
            const float* inPtr = buffer.getReadPointer (0);
            std::copy (inPtr, inPtr + numSamples, micMono.begin());
        }

        for (int ch = 0; ch < numOutChannels; ++ch)
            buffer.clear (ch, 0, numSamples);

        if (pendingAllNotesOff.exchange (false))
            stopActiveMelodyAndChords (midiMessages);

        // استخراج الخصائص الصوتية (RMS + ZCR + High/Low Energy + Peak) لتدريب البيت بوكس
        float sumSq = 0.0f, highEnergy = 0.0f, lowEnergy = 0.0f, peakAbs = 0.0f;
        int zeroCrossings = 0;
        for (int i = 0; i < numSamples; ++i)
        {
            float s = micMono[i];
            sumSq += s * s;
            peakAbs = juce::jmax (peakAbs, std::abs (s));

            // فلتر بسيط لفصل التردد المنخفض (دم/بيز) عن المرتفع (تك/سناير/هاي هات)
            lpFilterState = 0.88f * lpFilterState + 0.12f * s;
            float hp = s - lpFilterState;
            lowEnergy  += lpFilterState * lpFilterState;
            highEnergy += hp * hp;

            if (i > 0 && ((micMono[i - 1] >= 0.0f && s < 0.0f) || (micMono[i - 1] < 0.0f && s >= 0.0f)))
                ++zeroCrossings;

            yinBuffer[yinWritePos] = s;
            yinWritePos = (yinWritePos + 1) % kYinWindowSize;
        }

        float rms = std::sqrt (sumSq / (float) juce::jmax (1, numSamples));
        currentInputRms.store (rms);
        float zcr = (float) zeroCrossings / (float) juce::jmax (1, numSamples);
        float highRatio = highEnergy / (lowEnergy + highEnergy + 1.0e-6f);
        float sustainRatio = (peakAbs > 1.0e-5f) ? (rms / peakAbs) : 0.0f;

        vowelEnv = 0.85f * vowelEnv + 0.15f * juce::jlimit (0.0f, 1.0f, rms * 8.5f);

        const int mode = exclusiveMode.load();
        int manualPad = pendingPadTrigger.exchange (-1);
        if (manualPad >= 0 && manualPad < 8)
            fireTriggerPad (manualPad, midiMessages);

        // ======================================================================
        // وضع الكيك والسنير والبيت بوكس المعزول (ModeTriggersOnly)
        // ======================================================================
        if (mode == ModeTriggersOnly)
        {
            if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty())
                stopActiveMelodyAndChords (midiMessages);

            if (triggerCooldownSamples > 0)
                triggerCooldownSamples -= numSamples;

            float onsetDelta = rms - prevRms;
            prevRms = rms;
            float onsetThresh = 0.16f - (triggerSensitivity * 0.125f);

            if (onsetDelta > onsetThresh && triggerCooldownSamples <= 0 && rms > 0.02f)
            {
                int trainIdx = trainingPadIndex.load();
                if (trainIdx >= 0 && trainIdx < 8)
                {
                    // نحن في وضع تدريب الصوت: نلتقط بصمة الصوت (3 مرات) ونحفظها للـ Pad
                    auto& tp = pads[trainIdx];
                    float n = (float) tp.trainedHits;
                    tp.featZCR       = (tp.featZCR * n + zcr) / (n + 1.0f);
                    tp.featHighRatio = (tp.featHighRatio * n + highRatio) / (n + 1.0f);
                    tp.featSustain   = (tp.featSustain * n + sustainRatio) / (n + 1.0f);
                    tp.trainedHits++;
                    tp.isVoiceTrained = true;
                    fireTriggerPad (trainIdx, midiMessages);

                    if (tp.trainedHits >= 3)
                        trainingPadIndex.store (-1); // اكتمل التدريب (3/3)
                }
                else
                {
                    // البحث عن أقرب Pad مدرب يطابق بصمة صوت المستخدم الحالية
                    int bestPad = -1;
                    float bestDist = 999.0f;
                    for (int p = 0; p < 8; ++p)
                    {
                        if (pads[p].isVoiceTrained)
                        {
                            float dz = (zcr - pads[p].featZCR) * 3.2f;
                            float dh = (highRatio - pads[p].featHighRatio) * 2.5f;
                            float ds = (sustainRatio - pads[p].featSustain) * 1.4f;
                            float dist = std::sqrt (dz * dz + dh * dh + ds * ds);
                            if (dist < bestDist)
                            {
                                bestDist = dist;
                                bestPad = p;
                            }
                        }
                    }

                    // إذا لم يدرب المستخدم أي Pad بعد، نستخدم التقسيم الترددي الافتراضي (Pad 0, 1, 2)
                    if (bestPad < 0)
                        bestPad = (highRatio < 0.25f) ? 0 : ((zcr < 0.18f) ? 1 : 2);

                    fireTriggerPad (bestPad, midiMessages);
                }

                triggerCooldownSamples = (int) (currentSampleRate * 0.12);
            }
        }
        else
        {
            prevRms = rms;

            // ==================================================================
            // وضع الميلودي (0) أو وضع الكوردات (2) مع تحويل الأوكتاف والـ Semitone
            // ==================================================================
            if (rms > gateThreshold)
            {
                vowelEEE = juce::jlimit (0.0f, 1.0f, zcr * 3.5f);
                vowelOOO = juce::jlimit (0.0f, 1.0f, 1.0f - zcr * 2.8f);
                vowelAAA = juce::jlimit (0.0f, 1.0f, 1.0f - std::abs (zcr - 0.15f) * 4.0f);
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

                // 1) تطبيق النقل بالسيميتون (Semitone Transpose -12 .. +12)
                rawMidi += (float) semitoneTranspose;

                // 2) تثبيت أوكتاف البداية (Target Start Octave: Oct 1 .. Oct 5) أو اتباع الأوكتاف الطبيعي
                int candidateMidi = (int) std::round (rawMidi);
                if (targetStartOctave >= 1 && targetStartOctave <= 5 && manualNote < 0)
                {
                    int noteInScale = ((candidateMidi - rootNote) % 12 + 12) % 12;
                    int baseMidi = (targetStartOctave + 1) * 12 + rootNote; // C1=24, C2=36, C3=48, C4=60, C5=72
                    candidateMidi = baseMidi + noteInScale;
                }
                else
                {
                    candidateMidi += octaveShift * 12;
                }

                candidateMidi = juce::jlimit (12, 115, candidateMidi);

                float hysteresisWidth = 0.48f + (stickiness * 0.45f);
                if (activeRawMidiNote >= 0 && std::abs ((float) candidateMidi - (float) activeRawMidiNote) < hysteresisWidth)
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
                    float maxBendCents = (float) pbRangeSemitones * 100.0f;
                    int pbValue = 8192 + (int) std::round ((targetCents / maxBendCents) * 8192.0f);
                    pbValue = juce::jlimit (0, 16383, pbValue);
                    currentPitchBendValue.store (pbValue);
                    currentCentsOffset.store ((int) std::round (targetCents));

                    if (snappedMidi != activeOutputMidiNote)
                    {
                        stopActiveMelodyAndChords (midiMessages);
                        juce::uint8 vel = (juce::uint8) juce::jlimit (65, 127, (int) (vowelEnv * 115.0f + 50.0f));
                        activeOutputMidiNote = snappedMidi;
                        currentDetectedMidiNote.store (snappedMidi);

                        if (mode == ModeMelodyOnly)
                        {
                            emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), midiMessages, 0);
                            emitMidiEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, vel), midiMessages, 0);
                        }
                        else if (mode == ModeChordsOnly)
                        {
                            triggerSmartOrientalChord (snappedMidi, vel, midiMessages);
                        }
                    }
                    else if (mode == ModeMelodyOnly)
                    {
                        emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), midiMessages, 0);
                    }
                }
            }
            else
            {
                if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty())
                    stopActiveMelodyAndChords (midiMessages);
            }
        }

        // مخرج الصوت الداخلي (يشغل فقط الخامات التي حملها المستخدم على الـ Pads أو السينث الداخلي)
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
                            float v = samp[pad.playhead++] * 0.65f;
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

    void fireTriggerPad (int padIdx, juce::MidiBuffer& midiMessages)
    {
        auto& pad = pads[padIdx];
        pad.flashAlpha = 1.0f;
        if (pad.hasCustomSample)
            pad.playhead = 0;

        emitMidiEvent (juce::MidiMessage::noteOn  (triggersMidiChannel, pad.midiNote, (juce::uint8) 118), midiMessages, 0);
        emitMidiEvent (juce::MidiMessage::noteOff (triggersMidiChannel, pad.midiNote), midiMessages, 64);
    }

    void triggerSmartOrientalChord (int rootMidi, juce::uint8 vel, juce::MidiBuffer& midiMessages)
    {
        stopActiveChord (midiMessages);
        int pc = ((rootMidi % 12) + 12) % 12;
        if (noteStates[pc] == 2)
            activeChordNotes = { rootMidi, rootMidi + 7, rootMidi + 12 };
        else
        {
            int thirdPc = ((rootMidi + 4) % 12 + 12) % 12;
            int thirdInterval = (noteStates[thirdPc] == 1) ? 4 : 3;
            activeChordNotes = { rootMidi, rootMidi + thirdInterval, rootMidi + 7, rootMidi - 12 };
        }
        for (int n : activeChordNotes)
            emitMidiEvent (juce::MidiMessage::noteOn (chordsMidiChannel, n, (juce::uint8) (vel * 0.82f)), midiMessages, 0);
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
        state.setProperty ("targetStartOctave", targetStartOctave, nullptr);
        state.setProperty ("semitoneTranspose", semitoneTranspose, nullptr);
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
            targetStartOctave = state.getProperty ("targetStartOctave", 0);
            semitoneTranspose = state.getProperty ("semitoneTranspose", 0);
            applyMaqamPreset (rootNote, currentMaqamIndex);
        }
    }

    std::array<TriggerPadData, 8> pads;
    std::array<int, 12> noteStates { 1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2 };

    int rootNote = 0;
    int currentMaqamIndex = 2; // C Rast
    float quarterToneCents = -50.0f;
    float stickiness = 0.65f;
    float gateThreshold = 0.018f;
    float triggerSensitivity = 0.75f;
    int noteHoldFrames = 2;
    int pbRangeSemitones = 2;

    // إعدادات الأوكتاف والنقل الجديدة
    int targetStartOctave = 0; // 0 = Follow Voice, 1..5 = Lock output to Octave 1..5
    int semitoneTranspose = 0; // -12 .. +12 Semitones
    int octaveShift = 0;

    bool truBendMode = false;
    int synthPresetIndex = 0;
    int pitchMidiChannel = 1, chordsMidiChannel = 1, triggersMidiChannel = 1;

    float vowelAAA = 0.45f, vowelEEE = 0.30f, vowelOOO = 0.60f, vowelEnv = 0.50f;

    std::atomic<int>  exclusiveMode { ModeMelodyOnly };
    std::atomic<int>  trainingPadIndex { -1 };
    std::atomic<bool> pendingAllNotesOff { false };
    std::atomic<bool> isCapturingMidi { false };
    std::atomic<int>  capturedNoteCount { 0 };
    std::atomic<bool> intellAudioMuted { false };
    std::atomic<bool> trigAudioMuted   { false };
    std::atomic<int>  currentDetectedMidiNote { -1 };
    std::atomic<int>  currentPitchBendValue { 8192 };
    std::atomic<int>  currentCentsOffset { 0 };
    std::atomic<float> currentInputRms { 0.0f };
    std::atomic<int>  guiManualMidiNote { -1 };
    std::atomic<int>  pendingPadTrigger { -1 };

    juce::String directMidiDeviceName { "Click to Select loopMIDI" };

private:
    static constexpr int kYinWindowSize = 1024;
    std::vector<float> yinBuffer;
    int yinWritePos = 0;
    double currentSampleRate = 44100.0;
    double synthPhase = 0.0;
    float lpFilterState = 0.0f;

    int activeRawMidiNote = -1, activeOutputMidiNote = -1, pendingCandidateNote = -1, confirmCounter = 0;
    float prevRms = 0.0f;
    int triggerCooldownSamples = 0;

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

// ==============================================================================
// الواجهة الرسومية (VoxN2AudioProcessorEditor)
// ==============================================================================
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
            if (pad.flashAlpha > 0.01f)
                pad.flashAlpha *= 0.82f;
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
            "Load Custom Kick / Snare / Loop (.wav, .mp3)", juce::File{}, "*.wav;*.mp3;*.flac;*.ogg");
        fileChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this, padIndex] (const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file.existsAsFile()) { proc.loadCustomSampleForPad (padIndex, file); repaint(); }
            });
    }

    void paint (juce::Graphics& g) override
    {
        const juce::Colour bgDark (0xff1b1b1f), cardDark (0xff25252b), cyberGold (0xffe5b84b);
        const juce::Colour neonPink (0xfff03e76), neonGreen (0xff2ecc71), textBright (0xfff4f0e8), textMuted (0xff8e8d96);

        g.fillAll (bgDark);
        g.setColour (juce::Colour (0xff141417));
        g.fillRect (0, 0, getWidth(), 42);

        g.setColour (cyberGold);
        g.drawEllipse (12.0f, 10.0f, 22.0f, 22.0f, 2.0f);
        g.setFont (juce::Font (14.5f, juce::Font::bold));
        g.setColour (textBright);
        g.drawText ("VOXN 2", 38, 8, 60, 26, juce::Justification::centredLeft);

        // أزرار العزل الحصري الثلاثة
        const int activeMode = proc.exclusiveMode.load();
        const char* modeNames[3] = { "1. MELODY ONLY", "2. TRIGGERS ONLY", "3. CHORDS ONLY" };
        for (int m = 0; m < 3; ++m)
        {
            exclusiveModeRects[m] = { 105.0f + m * 124.0f, 8.0f, 118.0f, 26.0f };
            bool isCurr = (activeMode == m);
            g.setColour (isCurr ? (m == 1 ? neonPink : cyberGold) : juce::Colour (0xff23232a));
            g.fillRoundedRectangle (exclusiveModeRects[m], 5.0f);
            g.setColour (isCurr ? juce::Colours::black : textBright);
            g.setFont (juce::Font (10.5f, juce::Font::bold));
            g.drawText (modeNames[m], exclusiveModeRects[m], juce::Justification::centred);
        }

        // شريط MIDI Capture & Drag-to-DAW
        bool isRec = proc.isCapturingMidi.load();
        int noteCnt = proc.capturedNoteCount.load();

        midiRecButtonRect = { 482.0f, 8.0f, 112.0f, 26.0f };
        g.setColour (isRec ? neonPink : juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (midiRecButtonRect, 5.0f);
        g.setColour (textBright);
        g.drawText (isRec ? ("STOP [" + juce::String (noteCnt) + "]") : "● REC MIDI", midiRecButtonRect, juce::Justification::centred);

        midiDragButtonRect = { 600.0f, 8.0f, 185.0f, 26.0f };
        g.setColour (noteCnt > 0 ? neonGreen : cyberGold);
        g.fillRoundedRectangle (midiDragButtonRect, 5.0f);
        g.setColour (juce::Colours::black);
        g.drawText ("DRAG MIDI TO DAW (" + juce::String (noteCnt) + ")", midiDragButtonRect, juce::Justification::centred);

        midiClearButtonRect = { 792.0f, 8.0f, 65.0f, 26.0f };
        g.setColour (juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (midiClearButtonRect, 5.0f);
        g.setColour (textBright);
        g.drawText ("CLEAR", midiClearButtonRect, juce::Justification::centred);

        // التبويبات الخمسة
        const char* tabTitles[5] = { "Play", "Train (Beatbox)", "Key (Maqam/Scale)", "Chords", "Assign (MIDI)" };
        for (int i = 0; i < 5; ++i)
        {
            tabRects[i] = { 130.0f + i * 144.0f, 46.0f, 138.0f, 30.0f };
            bool isCurr = (currentTab == i);
            g.setColour (isCurr ? cardDark : juce::Colour (0xff19191e));
            g.fillRoundedRectangle (tabRects[i], 6.0f);
            g.setColour (isCurr ? cyberGold : textMuted);
            g.setFont (juce::Font (12.0f, juce::Font::bold));
            g.drawText (tabTitles[i], tabRects[i], juce::Justification::centred);
        }

        juce::Rectangle<float> mainCard (28.0f, 74.0f, 920.0f, 336.0f);
        g.setColour (cardDark);
        g.fillRoundedRectangle (mainCard, 10.0f);
        g.setColour (cyberGold.withAlpha (0.28f));
        g.drawRoundedRectangle (mainCard, 10.0f, 1.2f);

        if      (currentTab == 0) drawPlayTab   (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 1) drawTrainTab  (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 2) drawKeyTab    (g, cyberGold, neonPink, textBright, textMuted);
        else if (currentTab == 3) drawChordsTab (g, cyberGold, neonPink, textBright);
        else if (currentTab == 4) drawAssignTab (g, cyberGold, neonPink, neonGreen, textBright, textMuted);

        drawBottomSection (g, cyberGold, neonPink, neonGreen, textBright);
    }

    void drawPlayTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        const bool triggersActive = (proc.exclusiveMode.load() == ModeTriggersOnly);
        g.setColour (triggersActive ? pink : gold);
        g.setFont (juce::Font (11.5f, juce::Font::bold));
        g.drawText ("FPC DRUM PADS (EMPTY BY DEFAULT - DROP WAV/MP3):", 44, 80, 310, 22, juce::Justification::centredLeft);

        for (int i = 0; i < 8; ++i)
        {
            int row = i / 3, col = i % 3;
            juce::Rectangle<float> r (44.0f + col * 92.0f, 108.0f + row * 96.0f, 84.0f, 88.0f);
            padBounds[i] = r;
            auto& pad = proc.pads[i];

            g.setColour (juce::Colour (0xff1d1d22).interpolatedWith (pink, pad.flashAlpha * 0.5f));
            g.fillRoundedRectangle (r, 7.0f);
            g.setColour (pad.hasCustomSample ? gold : muted.withAlpha (0.4f));
            g.drawRoundedRectangle (r, 7.0f, 1.4f);

            padLoadRects[i] = { r.getX() + 4.0f, r.getY() + 4.0f, 34.0f, 16.0f };
            g.setColour (gold.withAlpha (0.25f));
            g.fillRoundedRectangle (padLoadRects[i], 3.0f);
            g.setColour (gold);
            g.setFont (9.0f);
            g.drawText ("LOAD", padLoadRects[i], juce::Justification::centred);

            // شارة هل المربع مدرب صوتياً أم لا
            if (pad.isVoiceTrained)
            {
                g.setColour (green);
                g.drawText ("MIC OK", (int) r.getRight() - 42, (int) r.getY() + 4, 38, 16, juce::Justification::centredRight);
            }

            g.setColour (bright);
            g.setFont (juce::Font (11.0f, juce::Font::bold));
            g.drawText (pad.name, (int) r.getX() + 5, (int) r.getY() + 36, 74, 18, juce::Justification::centredLeft);

            g.setColour (pad.hasCustomSample ? gold : muted);
            g.setFont (9.5f);
            g.drawText (pad.hasCustomSample ? pad.customFileName : "[Empty Pad]",
                        (int) r.getX() + 5, (int) r.getY() + 58, 74, 18, juce::Justification::centredLeft);
        }

        drawPitchRadarWheel (g, 495.0f, 245.0f, 120.0f, gold, pink, bright);

        g.setColour (muted);
        g.setFont (11.5f);
        g.drawText ("Stickiness:", 680, 82, 80, 20, juce::Justification::centredLeft);
        stickinessSliderRect = { 760.0f, 86.0f, 160.0f, 12.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (stickinessSliderRect, 6.0f);
        g.setColour (pink);
        g.fillRoundedRectangle (stickinessSliderRect.getX(), stickinessSliderRect.getY(),
                                stickinessSliderRect.getWidth() * proc.stickiness, 12.0f, 6.0f);
    }

    void drawPitchRadarWheel (juce::Graphics& g, float cx, float cy, float radius,
                              juce::Colour gold, juce::Colour pink, juce::Colour bright)
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
                               a1 + juce::MathConstants<float>::halfPi, a2 + juce::MathConstants<float>::halfPi, 0.56f);

            g.setColour (state == 0 ? juce::Colour (0xff18181c) : (isPlaying ? (state == 2 ? gold : pink) : (state == 2 ? gold.withAlpha (0.3f) : juce::Colour (0xff32323b))));
            g.fillPath (seg);

            float midRad = juce::degreesToRadians (angleDeg);
            float tx = cx + std::cos (midRad) * (radius * 0.78f);
            float ty = cy + std::sin (midRad) * (radius * 0.78f);
            wheelSegmentCenters[i] = { tx, ty };

            if (state > 0)
            {
                g.setColour (isPlaying ? juce::Colours::black : bright);
                g.setFont (juce::Font (11.0f, juce::Font::bold));
                g.drawText (juce::String (kNoteNames[i]) + (state == 2 ? " 1/2b" : ""), (int) tx - 22, (int) ty - 10, 44, 20, juce::Justification::centred);
            }
        }

        g.setColour (juce::Colour (0xff19191f));
        g.fillEllipse (cx - 46.0f, cy - 46.0f, 92.0f, 92.0f);
        g.setColour (gold);
        g.drawEllipse (cx - 46.0f, cy - 46.0f, 92.0f, 92.0f, 2.0f);

        juce::String centerNote = (activeMidi >= 0) ? (juce::String (kNoteNames[activePc]) + juce::String (activeMidi / 12 - 1)) : "READY";
        g.setColour (bright);
        g.setFont (juce::Font (15.0f, juce::Font::bold));
        g.drawText (centerNote, (int) cx - 40, (int) cy - 12, 80, 24, juce::Justification::centred);
    }

    // ==========================================================================
    // نافذة TRAIN: مدرب الصوت التفاعلي السهل (3 ضربات فقط) + تحميل الخامات
    // ==========================================================================
    void drawTrainTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText ("1. SELECT A PAD   ->   2. LOAD YOUR WAV/MP3   ->   3. TRAIN YOUR BEATBOX VOICE (3 HITS):", 48, 84, 760, 22, juce::Justification::centredLeft);

        for (int i = 0; i < 8; ++i)
        {
            juce::Rectangle<float> r (48.0f + (i % 2) * 150.0f, 116.0f + (i / 2) * 66.0f, 140.0f, 56.0f);
            padBounds[i] = r;
            bool sel = (selectedTrainPad == i);
            g.setColour (sel ? gold.withAlpha (0.22f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (r, 6.0f);
            g.setColour (sel ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (r, 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::Font (12.0f, juce::Font::bold));
            g.drawText (proc.pads[i].name + (proc.pads[i].isVoiceTrained ? " [TRAINED]" : ""), r.reduced (8.0f), juce::Justification::topLeft);

            g.setColour (proc.pads[i].hasCustomSample ? gold : muted);
            g.setFont (10.5f);
            g.drawText (proc.pads[i].hasCustomSample ? proc.pads[i].customFileName : "[Empty - No Sample]", r.reduced (8.0f), juce::Justification::bottomLeft);
        }

        // دائرة حالة التدريب الصوتي في المنتصف (3 خطوات واضحة)
        auto& currPad = proc.pads[selectedTrainPad];
        bool isListeningNow = (proc.trainingPadIndex.load() == selectedTrainPad);
        float cx = 505.0f, cy = 242.0f;

        g.setColour (juce::Colour (0xff191920));
        g.fillEllipse (cx - 88.0f, cy - 88.0f, 176.0f, 176.0f);
        g.setColour (isListeningNow ? pink : (currPad.isVoiceTrained ? green : gold));
        g.drawEllipse (cx - 88.0f, cy - 88.0f, 176.0f, 176.0f, 3.5f);

        g.setColour (bright);
        g.setFont (juce::Font (14.0f, juce::Font::bold));
        juce::String statusText = isListeningNow
            ? ("SPEAK NOW!\nHit " + juce::String (currPad.trainedHits) + " / 3")
            : (currPad.isVoiceTrained ? "VOICE TRAINED!\n(Ready)" : "NOT TRAINED\nClick Train ->");
        g.drawFittedText (statusText, (int) cx - 70, (int) cy - 30, 140, 60, juce::Justification::centred, 2);

        // أزرار التحكم على اليمين
        trainVoiceBtnRect  = { 650.0f, 125.0f, 270.0f, 44.0f };
        trainLoadButtonRect = { 650.0f, 182.0f, 270.0f, 40.0f };
        trainPlayButtonRect = { 650.0f, 234.0f, 270.0f, 40.0f };
        trainRevertButtonRect = { 650.0f, 286.0f, 270.0f, 40.0f };

        g.setColour (isListeningNow ? green : pink);
        g.fillRoundedRectangle (trainVoiceBtnRect, 7.0f);
        g.setColour (bright);
        g.setFont (juce::Font (12.5f, juce::Font::bold));
        g.drawText (isListeningNow ? "LISTENING... MAKE SOUND 3 TIMES!" : "🎤 TRAIN VOICE (MAKE SOUND 3X)", trainVoiceBtnRect, juce::Justification::centred);

        g.setColour (gold);
        g.fillRoundedRectangle (trainLoadButtonRect, 7.0f);
        g.setColour (juce::Colours::black);
        g.drawText ("📁 LOAD SAMPLE (.WAV / .MP3)", trainLoadButtonRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff32323b));
        g.fillRoundedRectangle (trainPlayButtonRect, 7.0f);
        g.setColour (bright);
        g.drawText ("▶ TEST / PLAY PAD SOUND", trainPlayButtonRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff292226));
        g.fillRoundedRectangle (trainRevertButtonRect, 7.0f);
        g.setColour (bright);
        g.drawText ("🗑️ CLEAR PAD SAMPLE & TRAINING", trainRevertButtonRect, juce::Justification::centred);
    }

    // ==========================================================================
    // نافذة KEY: تضم Major و Minor + المقامات الشرقية + إعدادات الأوكتاف
    // ==========================================================================
    void drawKeyTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright, juce::Colour muted)
    {
        drawPitchRadarWheel (g, 200.0f, 242.0f, 115.0f, gold, pink, bright);

        g.setColour (gold);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText ("SCALES & MAQAMAT (MAJOR, MINOR & ORIENTAL):", 365, 84, 520, 22, juce::Justification::centredLeft);

        for (int m = 0; m < 10; ++m)
        {
            maqamPresetRects[m] = { 365.0f + (m % 2) * 275.0f, 112.0f + (m / 2) * 44.0f, 260.0f, 38.0f };
            bool active = (proc.currentMaqamIndex == m);
            g.setColour (active ? gold.withAlpha (0.28f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (maqamPresetRects[m], 6.0f);
            g.setColour (active ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (maqamPresetRects[m], 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::Font (12.5f, juce::Font::bold));
            g.drawText (juce::String (kMaqamat[m].nameEn) + "  (" + juce::String::fromUTF8 (kMaqamat[m].nameAr) + ")",
                        maqamPresetRects[m].reduced (10.0f), juce::Justification::centredLeft);
        }

        g.setColour (bright);
        g.setFont (12.0f);
        g.drawText ("Quarter-Tone Depth: " + juce::String ((int) proc.quarterToneCents) + " cents", 365, 342, 230, 22, juce::Justification::centredLeft);

        qtDepthSliderRect = { 595.0f, 346.0f, 290.0f, 14.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (qtDepthSliderRect, 7.0f);
        float normQt = (proc.quarterToneCents - (-65.0f)) / 30.0f;
        g.setColour (gold);
        g.fillRoundedRectangle (qtDepthSliderRect.getX(), qtDepthSliderRect.getY(),
                                qtDepthSliderRect.getWidth() * juce::jlimit (0.0f, 1.0f, normQt), 14.0f, 7.0f);
    }

    void drawChordsTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour bright)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.5f, juce::Font::bold));
        g.drawText ("SMART ORIENTAL & WESTERN CHORDS (AUTO 5TH POWER ON QUARTER-TONES):", 48, 88, 660, 22, juce::Justification::centredLeft);

        const char* roman[7] = { "I", "II", "III", "IV", "V", "VI", "VII" };
        int degIdx = 0;
        for (int pc = 0; pc < 12 && degIdx < 7; ++pc)
        {
            int note = (proc.rootNote + pc) % 12;
            if (proc.noteStates[note] > 0)
            {
                juce::Rectangle<float> col (48.0f + degIdx * 124.0f, 140.0f, 114.0f, 220.0f);
                bool isQt = (proc.noteStates[note] == 2);
                g.setColour (juce::Colour (0xff1b1b20));
                g.fillRoundedRectangle (col, 7.0f);
                g.setColour (isQt ? gold : pink);
                g.drawRoundedRectangle (col, 7.0f, 1.4f);

                g.setColour (bright);
                g.setFont (juce::Font (14.0f, juce::Font::bold));
                g.drawText (juce::String (roman[degIdx]) + " - " + kNoteNames[note], col.removeFromTop (40.0f), juce::Justification::centred);
                g.setColour (isQt ? gold : bright);
                g.drawText (isQt ? "5th Power\n(Quarter-Safe)" : "Scale Triad\n+ Bassline", col, juce::Justification::centred);
                ++degIdx;
            }
        }
    }

    void drawAssignTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::Font (13.5f, juce::Font::bold));
        g.drawText ("DIRECT HARDWARE MIDI EMULATION (RECORD LIVE IN FL STUDIO):", 48, 88, 680, 22, juce::Justification::centredLeft);

        directMidiBtnRect = { 48.0f, 130.0f, 420.0f, 44.0f };
        g.setColour (green);
        g.fillRoundedRectangle (directMidiBtnRect, 7.0f);
        g.setColour (juce::Colours::black);
        g.setFont (juce::Font (13.0f, juce::Font::bold));
        g.drawText ("DIRECT MIDI OUT: " + proc.directMidiDeviceName + " (CLICK)", directMidiBtnRect, juce::Justification::centred);

        g.setColour (bright);
        g.setFont (13.0f);
        g.drawText ("1. Run loopMIDI and enable 'loopMIDI Port' in FL Studio -> Options -> MIDI Settings -> Input.", 48, 195, 780, 24, juce::Justification::centredLeft);
        g.drawText ("2. Or click [REC MIDI] in the top bar, sing, and drag [DRAG MIDI TO DAW] directly into FL Studio's Piano Roll!", 48, 225, 780, 24, juce::Justification::centredLeft);
    }

    // ==========================================================================
    // القسم السفلي: اختيار المقام + تحديد أوكتاف البداية (Out Oct) + Semitone (-12..+12)
    // ==========================================================================
    void drawBottomSection (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright)
    {
        rootCycleRect = { 114.0f, 414.0f, 68.0f, 25.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (rootCycleRect, 5.0f);
        g.setColour (gold);
        g.setFont (juce::Font (11.5f, juce::Font::bold));
        g.drawText ("Key: " + juce::String (kNoteNames[proc.rootNote]), rootCycleRect, juce::Justification::centred);

        maqamCycleRect = { 188.0f, 414.0f, 130.0f, 25.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (maqamCycleRect, 5.0f);
        g.setColour (gold);
        g.drawText ("Scale: " + juce::String (kMaqamat[proc.currentMaqamIndex].nameEn), maqamCycleRect, juce::Justification::centred);

        // زر تحديد الأوكتاف الذي يبدأ منه الخرج (Out Oct: Follow أو Lock Oct 1..5)
        startOctaveRect = { 326.0f, 414.0f, 175.0f, 25.0f };
        g.setColour (proc.targetStartOctave > 0 ? pink : juce::Colour (0xff26262e));
        g.fillRoundedRectangle (startOctaveRect, 5.0f);
        g.setColour (bright);
        juce::String octText = (proc.targetStartOctave == 0)
            ? "Out Oct: Follow Voice"
            : ("Out Oct: LOCK C" + juce::String (proc.targetStartOctave) + " (Oct " + juce::String (proc.targetStartOctave) + ")");
        g.drawText (octText, startOctaveRect, juce::Justification::centred);

        // أزرار النقل بالسيميتون (-12 .. +12 ST)
        semiMinusRect = { 510.0f, 414.0f, 34.0f, 25.0f };
        semiDisplayRect = { 548.0f, 414.0f, 118.0f, 25.0f };
        semiPlusRect = { 670.0f, 414.0f, 34.0f, 25.0f };

        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (semiMinusRect, 5.0f);
        g.fillRoundedRectangle (semiDisplayRect, 5.0f);
        g.fillRoundedRectangle (semiPlusRect, 5.0f);
        g.setColour (gold);
        g.drawText ("-", semiMinusRect, juce::Justification::centred);
        g.drawText ("Transpose: " + juce::String (proc.semitoneTranspose > 0 ? "+" : "") + juce::String (proc.semitoneTranspose) + " ST", semiDisplayRect, juce::Justification::centred);
        g.drawText ("+", semiPlusRect, juce::Justification::centred);

        // عجلتي Pitch و Mod
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (38.0f, 446.0f, 24.0f, 102.0f, 5.0f);
        g.fillRoundedRectangle (70.0f, 446.0f, 24.0f, 102.0f, 5.0f);
        float pbNorm = (float) proc.currentPitchBendValue.load() / 16383.0f;
        g.setColour (pink);
        g.fillRoundedRectangle (40.0f, 446.0f + (1.0f - pbNorm) * 86.0f, 20.0f, 14.0f, 3.0f);
        g.setColour (gold);
        g.fillRoundedRectangle (72.0f, 446.0f + (1.0f - proc.vowelEnv) * 86.0f, 20.0f, 14.0f, 3.0f);

        // البيانو المتدرج (36 مفتاحاً)
        const float kbX = 114.0f, kbY = 446.0f, kbW = 830.0f, kbH = 102.0f;
        const int numKeys = 36;
        const float keyW = kbW / (float) numKeys;
        int activeMidi = proc.currentDetectedMidiNote.load();

        for (int k = 0; k < numKeys; ++k)
        {
            int midi = 48 + k, pc = midi % 12, state = proc.noteStates[pc];
            juce::Rectangle<float> kr (kbX + k * keyW, kbY, keyW - 1.5f, kbH);
            pianoKeyRects[k] = kr;

            g.setColour (midi == activeMidi ? bright : (state == 2 ? gold : (state == 1 ? pink.interpolatedWith (gold, (float) k / 36.0f * 0.45f) : juce::Colour (0xff232329))));
            g.fillRoundedRectangle (kr, 3.0f);
        }

        // الشريط السفلي
        bottomDirectMidiRect = { 38.0f, 560.0f, 290.0f, 28.0f };
        g.setColour (juce::Colour (0xff1f3328));
        g.fillRoundedRectangle (bottomDirectMidiRect, 5.0f);
        g.setColour (green);
        g.drawText ("HW MIDI: " + proc.directMidiDeviceName, bottomDirectMidiRect, juce::Justification::centred);

        synthCycleRect = { 340.0f, 560.0f, 195.0f, 28.0f };
        g.setColour (juce::Colour (0xff25252c));
        g.fillRoundedRectangle (synthCycleRect, 5.0f);
        g.setColour (gold);
        const char* synthNames[4] = { "Synth: Electric Saz", "Synth: G-Clarinet", "Synth: Electric Oud", "Synth: Sub Bass" };
        g.drawText (synthNames[proc.synthPresetIndex % 4], synthCycleRect, juce::Justification::centred);

        intellMuteRect = { 555.0f, 560.0f, 145.0f, 28.0f };
        bool intMuted = proc.intellAudioMuted.load();
        g.setColour (intMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (intellMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (intMuted ? "Intell Audio: MUTED" : "Intell Audio: ON", intellMuteRect, juce::Justification::centred);

        trigMuteRect = { 710.0f, 560.0f, 145.0f, 28.0f };
        bool trMuted = proc.trigAudioMuted.load();
        g.setColour (trMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (trigMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (trMuted ? "Trig Audio: MUTED" : "Trig Audio: ON", trigMuteRect, juce::Justification::centred);
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        juce::Point<float> pt = e.position;

        if (bottomDirectMidiRect.contains (pt) || (currentTab == 4 && directMidiBtnRect.contains (pt)))
        {
            proc.cycleDirectMidiDevice(); repaint(); return;
        }

        for (int m = 0; m < 3; ++m)
            if (exclusiveModeRects[m].contains (pt)) { proc.setExclusiveEngineMode (m); repaint(); return; }

        if (midiRecButtonRect.contains (pt))
        {
            if (proc.isCapturingMidi.load()) proc.stopMidiCapture();
            else proc.startMidiCapture();
            repaint(); return;
        }
        if (midiClearButtonRect.contains (pt)) { proc.clearMidiCapture(); repaint(); return; }

        for (int i = 0; i < 5; ++i)
        {
            if (tabRects[i].contains (pt))
            {
                currentTab = i;
                if      (i == 1) proc.setExclusiveEngineMode (ModeTriggersOnly);
                else if (i == 2) proc.setExclusiveEngineMode (ModeMelodyOnly);
                else if (i == 3) proc.setExclusiveEngineMode (ModeChordsOnly);
                repaint(); return;
            }
        }

        if (rootCycleRect.contains (pt))   { proc.applyMaqamPreset ((proc.rootNote + 1) % 12, proc.currentMaqamIndex); repaint(); return; }
        if (maqamCycleRect.contains (pt))  { proc.applyMaqamPreset (proc.rootNote, (proc.currentMaqamIndex + 1) % (int) kMaqamat.size()); repaint(); return; }
        if (startOctaveRect.contains (pt)) { proc.targetStartOctave = (proc.targetStartOctave + 1) % 6; repaint(); return; }
        if (semiMinusRect.contains (pt))   { proc.semitoneTranspose = juce::jlimit (-24, 24, proc.semitoneTranspose - 1); repaint(); return; }
        if (semiPlusRect.contains (pt))    { proc.semitoneTranspose = juce::jlimit (-24, 24, proc.semitoneTranspose + 1); repaint(); return; }
        if (semiDisplayRect.contains (pt)) { proc.semitoneTranspose = 0; repaint(); return; }

        if (intellMuteRect.contains (pt))  { proc.intellAudioMuted.store (! proc.intellAudioMuted.load()); repaint(); return; }
        if (trigMuteRect.contains (pt))    { proc.trigAudioMuted.store (! proc.trigAudioMuted.load()); repaint(); return; }
        if (synthCycleRect.contains (pt))  { proc.synthPresetIndex = (proc.synthPresetIndex + 1) % 4; repaint(); return; }

        for (int k = 0; k < 36; ++k)
            if (pianoKeyRects[k].contains (pt)) { proc.setExclusiveEngineMode (ModeMelodyOnly); proc.guiManualMidiNote.store (48 + k); repaint(); return; }

        if (currentTab == 0)
        {
            for (int i = 0; i < 8; ++i)
            {
                if (padLoadRects[i].contains (pt)) { openSampleChooserForPad (i); return; }
                if (padBounds[i].contains (pt))    { proc.setExclusiveEngineMode (ModeTriggersOnly); proc.triggerPadNow (i); repaint(); return; }
            }
            if (stickinessSliderRect.contains (pt))
            {
                proc.stickiness = juce::jlimit (0.0f, 1.0f, (pt.x - stickinessSliderRect.getX()) / stickinessSliderRect.getWidth());
                repaint(); return;
            }
            for (int i = 0; i < 12; ++i)
            {
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 24.0f)
                {
                    proc.setExclusiveEngineMode (ModeMelodyOnly);
                    if (e.mods.isRightButtonDown() || e.mods.isShiftDown()) proc.noteStates[i] = (proc.noteStates[i] + 1) % 3;
                    else proc.guiManualMidiNote.store (60 + i);
                    repaint(); return;
                }
            }
        }
        else if (currentTab == 1) // TRAIN
        {
            for (int i = 0; i < 8; ++i)
                if (padBounds[i].contains (pt)) { selectedTrainPad = i; repaint(); return; }

            if (trainVoiceBtnRect.contains (pt))   { proc.startVoiceTrainingForPad (selectedTrainPad); repaint(); return; }
            if (trainLoadButtonRect.contains (pt)) { openSampleChooserForPad (selectedTrainPad); return; }
            if (trainPlayButtonRect.contains (pt)) { proc.triggerPadNow (selectedTrainPad); return; }
            if (trainRevertButtonRect.contains (pt))
            {
                proc.clearPadSample (selectedTrainPad);
                proc.clearVoiceTrainingForPad (selectedTrainPad);
                repaint(); return;
            }
        }
        else if (currentTab == 2) // KEY
        {
            for (int m = 0; m < 10; ++m)
                if (maqamPresetRects[m].contains (pt)) { proc.applyMaqamPreset (proc.rootNote, m); repaint(); return; }

            if (qtDepthSliderRect.contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - qtDepthSliderRect.getX()) / qtDepthSliderRect.getWidth());
                proc.quarterToneCents = -65.0f + norm * 30.0f;
                repaint(); return;
            }
            for (int i = 0; i < 12; ++i)
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 24.0f)
                {
                    proc.noteStates[i] = (proc.noteStates[i] + 1) % 3;
                    repaint(); return;
                }
        }
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (! isDraggingMidi && midiDragButtonRect.contains (e.mouseDownPosition) && e.getDistanceFromDragStart() > 4)
        {
            isDraggingMidi = true;
            proc.stopMidiCapture();
            juce::File midiFile = proc.exportCapturedMidiToTempFile();
            juce::DragAndDropContainer::performExternalDragDropOfFiles ({ midiFile.getFullPathName() }, false, this, [this]() { isDraggingMidi = false; });
        }
    }

    void mouseUp (const juce::MouseEvent&) override
    {
        isDraggingMidi = false;
        proc.guiManualMidiNote.store (-1);
    }

private:
    VoxN2AudioProcessor& proc;
    int currentTab = 0;
    int selectedTrainPad = 0;
    bool isDraggingMidi = false;

    std::array<juce::Rectangle<float>, 3> exclusiveModeRects;
    std::array<juce::Rectangle<float>, 5> tabRects;
    std::array<juce::Rectangle<float>, 8> padBounds;
    std::array<juce::Rectangle<float>, 8> padLoadRects;
    std::array<juce::Point<float>, 12> wheelSegmentCenters;
    std::array<juce::Rectangle<float>, 10> maqamPresetRects;
    std::array<juce::Rectangle<float>, 36> pianoKeyRects;

    juce::Rectangle<float> midiRecButtonRect, midiDragButtonRect, midiClearButtonRect;
    juce::Rectangle<float> stickinessSliderRect, qtDepthSliderRect;
    juce::Rectangle<float> trainVoiceBtnRect, trainLoadButtonRect, trainPlayButtonRect, trainRevertButtonRect;
    juce::Rectangle<float> directMidiBtnRect, bottomDirectMidiRect;
    juce::Rectangle<float> rootCycleRect, maqamCycleRect, startOctaveRect, semiMinusRect, semiDisplayRect, semiPlusRect;
    juce::Rectangle<float> synthCycleRect, intellMuteRect, trigMuteRect;

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
