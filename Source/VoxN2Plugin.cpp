#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_dsp/juce_dsp.h>
#include "BinaryData.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <cmath>
#include <array>
#include <vector>

static const char* kPluginVersionString = "1.0.0";
static const char* kManufacturerName    = "Mido Mazzika";
// When your website/contact/buy link is ready, put it between the quotes below:
static const char* kManufacturerUrl     = "";

enum EngineExclusiveMode
{
    ModeMelodyOnly   = 0,
    ModeTriggersOnly = 1,
    ModeChordsOnly   = 2
};

enum MelodyVoiceMode
{
    VoiceMono   = 0,
    VoiceLegato = 1,
    VoicePoly   = 2
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

static const std::array<const char*, 4> kFactoryPresetNames = {{
    "1. Oriental Wind / Nay",
    "2. Clean Piano / Oud",
    "3. Smart Chords Pad",
    "4. Tight Beatbox Kit"
}};

struct ChordVoicingOption
{
    juce::String label;
    std::vector<int> offsets;
};

struct PolyVoice
{
    int note = -1;
    int samplesRemaining = 0;
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
    float levelTrim = 1.0f; // 0.15 .. 1.00 (controls outgoing MIDI velocity & internal sample level)
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
            pads[i].levelTrim = 1.0f;
        }

        pitchTraceMidi.fill (60.0f);
        pitchTraceTargetMidi.fill (60.0f);
        pitchTraceValid.fill (false);

        selectedDegreeVoicing.fill (0);
        applyMaqamPreset (0, 2);
        applyFactoryPreset (0);
    }

    ~VoxN2AudioProcessor() override = default;

    void prepareToPlay (double sampleRate, int) override
    {
        currentSampleRate = (sampleRate > 0.0) ? sampleRate : 44100.0;
        yinBuffer.assign (kYinWindowSize, 0.0f);
        yinWritePos = 0;
        samplesSinceLastYin = 0;
        lastDetectedFreq = -1.0f;
        lastPitchConfidence = 0.0f;

        synthPhase = 0.0;
        currentSynthFreq = 261.63;
        hadPreviousSynthNote = false;

        lpFilterState = 0.0f;
        hpCondenserState = 0.0f;
        prevSampleRaw = 0.0f;
        prevSamplePreEmp = 0.0f;

        pitchHpState = 0.0f;
        pitchPrevRaw = 0.0f;
        pitchLp1 = 0.0f;
        pitchLp2 = 0.0f;
        pitchLpAlpha = (float) (1.0 - std::exp (-juce::MathConstants<double>::twoPi * 1100.0 / currentSampleRate));
        pitchHpCoeff = (float) std::exp (-juce::MathConstants<double>::twoPi * 75.0 / currentSampleRate);

        pendingLegatoOffNote = -1;
        legatoOffSamplesRemaining = 0;
        polyMelodyVoices.clear();
        pendingCandidateMidi = -1;
        pendingCandidateSamples = 0;
        releaseHoldSamplesRemaining = 0;

        smoothedVibCents = 0.0f;
        lastSentPitchWheel = -1;
        lastSentModCC = -1;
        ccThrottleSamples = 0;
        pbThrottleSamples = 0;
        traceThrottleSamples = 0;
        pendingGlideCCUpdate.store (true);
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    // ==================== PRESETS SYSTEM ====================
    juce::File getUserPresetsDirectory() const
    {
        auto dir = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory)
                       .getChildFile (kManufacturerName)
                       .getChildFile ("VoxNI")
                       .getChildFile ("Presets");
        if (! dir.exists()) dir.createDirectory();
        return dir;
    }

    juce::Array<juce::File> getUserPresetFiles() const
    {
        return getUserPresetsDirectory().findChildFiles (juce::File::findFiles, false, "*.voxni");
    }

    void applyFactoryPreset (int index)
    {
        int idx = juce::jlimit (0, (int) kFactoryPresetNames.size() - 1, index);
        currentPresetIndex = idx;
        currentPresetName = kFactoryPresetNames[idx];

        if (idx == 0) // Oriental Wind / Nay
        {
            setExclusiveEngineMode (ModeMelodyOnly);
            setMelodyVoiceMode (VoiceLegato);
            setGlideMs (85.0f);
            melodyNoteLockMs = 22.0f;
            stickiness = 0.62f;
            vibratoEnabled.store (true);
            vibratoDepthMode.store (1); // Med
            sendAutomationCC.store (true);
            ccSensitivityMode.store (2); // High
            fixedVelocity127 = false;
            velocityCurve = 0.35f;
        }
        else if (idx == 1) // Clean Piano / Oud
        {
            setExclusiveEngineMode (ModeMelodyOnly);
            setMelodyVoiceMode (VoiceMono);
            setGlideMs (0.0f);
            melodyNoteLockMs = 35.0f;
            stickiness = 0.72f;
            vibratoEnabled.store (false);
            vibratoDepthMode.store (0);
            sendAutomationCC.store (true);
            ccSensitivityMode.store (0); // Soft
            fixedVelocity127 = false;
            velocityCurve = 0.20f;
        }
        else if (idx == 2) // Smart Chords Pad
        {
            setExclusiveEngineMode (ModeChordsOnly);
            chordLockMs = 85.0f;
            stickiness = 0.76f;
            vibratoEnabled.store (false);
            sendAutomationCC.store (true);
            ccSensitivityMode.store (0); // Soft
            fixedVelocity127 = false;
        }
        else if (idx == 3) // Tight Beatbox Kit
        {
            setExclusiveEngineMode (ModeTriggersOnly);
            triggerSensitivity = 0.88f;
            trainSensitivity = 0.90f;
            hiHatBoost = 0.90f;
            triggerRetrigLockMs = 75.0f;
            fixedVelocity127 = false;
            velocityCurve = 0.45f;
        }
    }

    juce::ValueTree buildStateValueTree() const
    {
        juce::ValueTree state ("VoxNIState");
        state.setProperty ("presetName", currentPresetName, nullptr);
        state.setProperty ("exclusiveMode", exclusiveMode.load(), nullptr);
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
        state.setProperty ("stickiness", stickiness, nullptr);
        state.setProperty ("melodyVoiceMode", melodyVoiceMode.load(), nullptr);
        state.setProperty ("glideMs", glideMs, nullptr);
        state.setProperty ("melodyNoteLockMs", melodyNoteLockMs, nullptr);
        state.setProperty ("chordLockMs", chordLockMs, nullptr);
        state.setProperty ("triggerRetrigLockMs", triggerRetrigLockMs, nullptr);
        state.setProperty ("triggerSensitivity", triggerSensitivity, nullptr);
        state.setProperty ("trainSensitivity", trainSensitivity, nullptr);
        state.setProperty ("hiHatBoost", hiHatBoost, nullptr);
        state.setProperty ("vibratoEnabled", vibratoEnabled.load(), nullptr);
        state.setProperty ("vibratoDepthMode", vibratoDepthMode.load(), nullptr);
        state.setProperty ("sendAutomationCC", sendAutomationCC.load(), nullptr);
        state.setProperty ("ccSensitivityMode", ccSensitivityMode.load(), nullptr);
        state.setProperty ("synthPresetIndex", synthPresetIndex, nullptr);

        for (int i = 0; i < 8; ++i)
        {
            state.setProperty ("padLvl" + juce::String (i), pads[i].levelTrim, nullptr);
            state.setProperty ("padSens" + juce::String (i), pads[i].padSensitivity, nullptr);
        }
        return state;
    }

    void applyStateValueTree (const juce::ValueTree& state)
    {
        if (! state.isValid()) return;
        currentPresetName = state.getProperty ("presetName", currentPresetName).toString();
        setExclusiveEngineMode ((int) state.getProperty ("exclusiveMode", exclusiveMode.load()));
        rootNote = (int) state.getProperty ("rootNote", 0);
        currentMaqamIndex = (int) state.getProperty ("maqamIndex", 2);
        quarterToneCents = (float) state.getProperty ("quarterToneCents", -50.0f);
        vocalRangePreset = (int) state.getProperty ("vocalRangePreset", 0);
        semitoneTranspose = (int) state.getProperty ("semitoneTranspose", 0);
        velocityCurve = (float) state.getProperty ("velocityCurve", 0.35f);
        fixedVelocity127 = (bool) state.getProperty ("fixedVelocity127", false);
        globalGateLevel = (float) state.getProperty ("globalGateLevel", 0.16f);
        condenserMode = (bool) state.getProperty ("condenserMode", true);
        quantizeMode = (int) state.getProperty ("quantizeMode", 2);
        stickiness = (float) state.getProperty ("stickiness", 0.65f);
        melodyVoiceMode.store ((int) state.getProperty ("melodyVoiceMode", (int) VoiceMono));
        glideMs = (float) state.getProperty ("glideMs", 0.0f);
        melodyNoteLockMs = (float) state.getProperty ("melodyNoteLockMs", 28.0f);
        chordLockMs = (float) state.getProperty ("chordLockMs", 75.0f);
        triggerRetrigLockMs = (float) state.getProperty ("triggerRetrigLockMs", 85.0f);
        triggerSensitivity = (float) state.getProperty ("triggerSensitivity", 0.86f);
        trainSensitivity = (float) state.getProperty ("trainSensitivity", 0.90f);
        hiHatBoost = (float) state.getProperty ("hiHatBoost", 0.85f);
        vibratoEnabled.store ((bool) state.getProperty ("vibratoEnabled", true));
        vibratoDepthMode.store (juce::jlimit (0, 2, (int) state.getProperty ("vibratoDepthMode", 1)));
        sendAutomationCC.store ((bool) state.getProperty ("sendAutomationCC", true));
        ccSensitivityMode.store (juce::jlimit (0, 2, (int) state.getProperty ("ccSensitivityMode", 1)));
        synthPresetIndex = (int) state.getProperty ("synthPresetIndex", 0);

        for (int i = 0; i < 8; ++i)
        {
            pads[i].levelTrim = juce::jlimit (0.15f, 1.0f, (float) state.getProperty ("padLvl" + juce::String (i), 1.0f));
            pads[i].padSensitivity = juce::jlimit (0.10f, 1.0f, (float) state.getProperty ("padSens" + juce::String (i), 0.84f));
        }

        pendingGlideCCUpdate.store (true);
        applyMaqamPreset (rootNote, currentMaqamIndex);
    }

    bool savePresetToFile (const juce::File& file)
    {
        currentPresetName = file.getFileNameWithoutExtension();
        currentPresetIndex = -1;
        auto state = buildStateValueTree();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        if (xml != nullptr)
            return xml->writeTo (file);
        return false;
    }

    bool loadPresetFromFile (const juce::File& file)
    {
        if (! file.existsAsFile()) return false;
        std::unique_ptr<juce::XmlElement> xml (juce::XmlDocument::parse (file));
        if (xml == nullptr) return false;
        auto state = juce::ValueTree::fromXml (*xml);
        if (! state.isValid()) return false;
        currentPresetIndex = -1;
        applyStateValueTree (state);
        currentPresetName = file.getFileNameWithoutExtension();
        return true;
    }

    // ==================== AUTO ROOT DETECTION (ROOT ONLY) ====================
    void toggleAutoRootDetection()
    {
        if (isDetectingKey.load())
        {
            finalizeAutoRootDetection();
        }
        else
        {
            const juce::SpinLock::ScopedLockType sl (autoKeyLock);
            pitchClassWeights.fill (0.0f);
            lastDetectedPitchClass = -1;
            totalDetectedVocalFrames = 0;
            autoKeySamplesElapsed = 0;
            autoRootStatusText = "LISTENING...";
            isDetectingKey.store (true);
        }
    }

    void finalizeAutoRootDetection()
    {
        isDetectingKey.store (false);
        const juce::SpinLock::ScopedLockType sl (autoKeyLock);

        if (totalDetectedVocalFrames < 6)
        {
            autoRootStatusText = "TOO SHORT";
            return;
        }

        // Strongly weight the final resting note (Tonic / Qarar)
        if (lastDetectedPitchClass >= 0 && lastDetectedPitchClass < 12)
            pitchClassWeights[lastDetectedPitchClass] *= 1.65f;

        int bestRoot = rootNote;
        float bestWeight = -1.0f;
        for (int r = 0; r < 12; ++r)
        {
            if (pitchClassWeights[r] > bestWeight)
            {
                bestWeight = pitchClassWeights[r];
                bestRoot = r;
            }
        }

        // Keep user's selected Maqam/Scale untouched; update Root only
        applyMaqamPreset (bestRoot, currentMaqamIndex);
        autoRootStatusText = "ROOT: " + juce::String (kNoteNames[bestRoot]);
    }

    void emitMidiEvent (const juce::MidiMessage& msg, juce::MidiBuffer& midiMessages, int sampleOffset = 0)
    {
        midiMessages.addEvent (msg, juce::jmax (0, sampleOffset));
        recordMidiEventInternal (msg);
    }

    void emitPitchWheelIfChanged (int pbValue, juce::MidiBuffer& midiMessages, int sampleOffset = 0, bool forceSend = false)
    {
        if (forceSend || std::abs (pbValue - lastSentPitchWheel) >= 6)
        {
            emitMidiEvent (juce::MidiMessage::pitchWheel (pitchMidiChannel, pbValue), midiMessages, sampleOffset);
            lastSentPitchWheel = pbValue;
        }
    }

    void setExclusiveEngineMode (int newMode)
    {
        int clamped = juce::jlimit (0, 2, newMode);
        if (exclusiveMode.exchange (clamped) != clamped)
            pendingAllNotesOff.store (true);
    }

    void setMelodyVoiceMode (int newVoiceMode)
    {
        int clamped = juce::jlimit (0, 2, newVoiceMode);
        if (melodyVoiceMode.exchange (clamped) != clamped)
        {
            pendingAllNotesOff.store (true);
            pendingGlideCCUpdate.store (true);
        }
    }

    void setGlideMs (float ms)
    {
        glideMs = juce::jlimit (0.0f, 300.0f, ms);
        pendingGlideCCUpdate.store (true);
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
        pendingGlideCCUpdate.store (true);
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
        if (! captureFirstEventReceived && msg.isNoteOn())
        {
            captureFirstEventReceived = true;
            capturedSamplesElapsed = 0;
        }

        double bpm = hostBpm.load();
        double rawSec = captureFirstEventReceived ? ((double) capturedSamplesElapsed / currentSampleRate) : 0.0;
        double compensatedSec = juce::jmax (0.0, rawSec - 0.015);
        double ticks = compensatedSec * (bpm / 60.0) * 960.0;

        if (msg.isNoteOn() || msg.isNoteOff())
        {
            if (quantizeMode == 1) ticks = std::round (ticks / 480.0) * 480.0;
            else if (quantizeMode == 2) ticks = std::round (ticks / 240.0) * 240.0;
        }

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
                                             .getChildFile ("VoxNI_Captured.mid"));
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

    float detectPitchYINDecimated (const std::vector<float>& ringBuf, int writePos, double sr, float& outConfidence)
    {
        constexpr int decLen = kYinWindowSize / 2;
        constexpr int W      = decLen / 2;
        const double effSr   = sr * 0.5;

        std::array<float, decLen> decBuf;
        for (int i = 0; i < decLen; ++i)
        {
            int idx0 = (writePos + i * 2) % kYinWindowSize;
            int idx1 = (idx0 + 1) % kYinWindowSize;
            decBuf[i] = 0.5f * (ringBuf[idx0] + ringBuf[idx1]);
        }

        const int minTau = juce::jmax (2, (int) (effSr / 1050.0));
        const int maxTau = juce::jmin (W - 2, (int) (effSr / 62.0));

        std::array<float, W> diff {};
        std::array<float, W> cmnd {};
        cmnd[0] = 1.0f;

        float runningSum = 0.0f;
        for (int tau = 1; tau <= maxTau + 1; ++tau)
        {
            float sum = 0.0f;
            for (int i = 0; i < W; ++i)
            {
                float d = decBuf[i] - decBuf[i + tau];
                sum += d * d;
            }
            diff[tau] = sum;
            runningSum += sum;
            cmnd[tau] = (runningSum > 1.0e-9f) ? (sum * (float) tau / runningSum) : 1.0f;
        }

        int bestTau = -1;
        float threshold = condenserMode ? 0.16f : 0.19f;

        for (int tau = minTau; tau < maxTau; ++tau)
        {
            if (cmnd[tau] < threshold)
            {
                while (tau + 1 < maxTau && cmnd[tau + 1] < cmnd[tau]) ++tau;
                bestTau = tau;
                break;
            }
        }

        if (bestTau < 0)
        {
            outConfidence = 0.0f;
            return -1.0f;
        }

        outConfidence = 1.0f - cmnd[bestTau];
        float s0 = cmnd[bestTau - 1], s1 = cmnd[bestTau], s2 = cmnd[bestTau + 1];
        float denom = 2.0f * (2.0f * s1 - s2 - s0);
        float betterTau = (float) bestTau + ((std::abs (denom) > 1e-6f) ? (s2 - s0) / denom : 0.0f);
        return (betterTau > 0.0f) ? (float) (effSr / betterTau) : -1.0f;
    }

    void pushPitchTracePoint (float liveMidi, float targetMidi, bool valid)
    {
        int idx = pitchTraceWriteIdx.load();
        pitchTraceMidi[idx] = liveMidi;
        pitchTraceTargetMidi[idx] = targetMidi;
        pitchTraceValid[idx] = valid;
        pitchTraceWriteIdx.store ((idx + 1) % kPitchTraceLen);
    }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int numSamples = buffer.getNumSamples();
        if (numSamples <= 0) return;

        const int numInChannels = getTotalNumInputChannels();
        const int numOutChannels = getTotalNumOutputChannels();
        const int maxSampleIdx = juce::jmax (0, numSamples - 1);

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

        if (pendingGlideCCUpdate.exchange (false))
        {
            bool glideOn = (glideMs > 1.0f);
            int cc5Val = juce::jlimit (0, 127, (int) std::round ((glideMs / 300.0f) * 127.0f));
            emitMidiEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, 65, glideOn ? 127 : 0), midiMessages, 0);
            emitMidiEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, 5, cc5Val), midiMessages, 0);
        }

        if (pendingLegatoOffNote >= 0)
        {
            legatoOffSamplesRemaining -= numSamples;
            if (legatoOffSamplesRemaining <= 0)
            {
                if (pendingLegatoOffNote != activeOutputMidiNote)
                    emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, pendingLegatoOffNote), midiMessages, 0);
                pendingLegatoOffNote = -1;
                legatoOffSamplesRemaining = 0;
            }
        }

        if (! polyMelodyVoices.empty())
        {
            for (auto it = polyMelodyVoices.begin(); it != polyMelodyVoices.end(); )
            {
                it->samplesRemaining -= numSamples;
                if (it->samplesRemaining <= 0)
                {
                    if (it->note != activeOutputMidiNote)
                        emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, it->note), midiMessages, 0);
                    it = polyMelodyVoices.erase (it);
                }
                else
                {
                    ++it;
                }
            }
        }

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

                pitchHpState = pitchHpCoeff * (pitchHpState + raw - pitchPrevRaw);
                pitchPrevRaw = raw;
                pitchLp1 += pitchLpAlpha * (pitchHpState - pitchLp1);
                pitchLp2 += pitchLpAlpha * (pitchLp1 - pitchLp2);

                yinBuffer[yinWritePos] = pitchLp2;
                yinWritePos = (yinWritePos + 1) % kYinWindowSize;
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

        // Expression / Mod sensitivity scaling (Soft / Med / High)
        const float ccGainTable[3] = { 5.5f, 8.8f, 13.5f };
        float ccGain = ccGainTable[juce::jlimit (0, 2, ccSensitivityMode.load())];
        vowelEnv = 0.82f * vowelEnv + 0.18f * juce::jlimit (0.0f, 1.0f, rms * ccGain);

        const int mode = exclusiveMode.load();
        int manualPad = pendingPadTrigger.exchange (-1);
        if (manualPad >= 0 && manualPad < 8) fireTriggerPad (manualPad, 115, midiMessages, 0, numSamples);

        float traceLiveMidi = 60.0f;
        float traceTargetMidi = 60.0f;
        bool traceIsValid = false;

        if (mode == ModeTriggersOnly)
        {
            if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty() || ! polyMelodyVoices.empty())
                stopActiveMelodyAndChords (midiMessages);

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
                int retrigLockSamples = (int) (currentSampleRate * (triggerRetrigLockMs / 1000.0f));

                if (isTrainingNow)
                {
                    auto& tp = pads[trainIdx];
                    float n = (float) tp.trainedHits;
                    tp.featZCR       = (tp.featZCR * n + zcr) / (n + 1.0f);
                    tp.featHighRatio = (tp.featHighRatio * n + highRatio) / (n + 1.0f);
                    tp.featSustain   = (tp.featSustain * n + sustainRatio) / (n + 1.0f);
                    tp.trainedHits++;
                    tp.isVoiceTrained = true;
                    fireTriggerPad (trainIdx, dynVel, midiMessages, firstTransientOffset, numSamples);

                    if (tp.trainedHits >= 3) trainingPadIndex.store (-1);
                    triggerCooldownSamples = juce::jmax (retrigLockSamples, (int) (currentSampleRate * 0.13));
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
                        fireTriggerPad (bestPad, dynVel, midiMessages, firstTransientOffset, numSamples);
                        triggerCooldownSamples = retrigLockSamples;
                    }
                    else if (bestPad < 0)
                    {
                        int fallbackPad = (highRatio > 0.65f || zcr > 0.20f) ? 2 : ((highRatio > 0.28f) ? 1 : 0);
                        fireTriggerPad (fallbackPad, dynVel, midiMessages, firstTransientOffset, numSamples);
                        triggerCooldownSamples = retrigLockSamples;
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
                pitchConfidence = 1.0f;
            }
            else if (globalGateIsOpen)
            {
                samplesSinceLastYin += numSamples;
                if (samplesSinceLastYin >= 128 || lastDetectedFreq < 0.0f)
                {
                    lastDetectedFreq = detectPitchYINDecimated (yinBuffer, yinWritePos, currentSampleRate, lastPitchConfidence);
                    samplesSinceLastYin = 0;
                }
                detectedFreq = lastDetectedFreq;
                pitchConfidence = lastPitchConfidence;
            }
            else
            {
                lastDetectedFreq = -1.0f;
                lastPitchConfidence = 0.0f;
            }

            bool validVocalPitch = (detectedFreq > 62.0f && detectedFreq < 1050.0f)
                                   && (! condenserMode || pitchConfidence > 0.65f || manualNote >= 0);

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
                        int pc = ((nearestNote % 12) + 12) % 12;

                        pitchClassWeights[pc] += 1.0f;
                        lastDetectedPitchClass = pc;
                        totalDetectedVocalFrames++;
                    }
                    if (autoKeySamplesElapsed >= (int) (currentSampleRate * 3.8) && totalDetectedVocalFrames >= 10)
                        shouldFinishNow = true;
                }
                if (shouldFinishNow) finalizeAutoRootDetection();
            }

            if (validVocalPitch)
            {
                releaseHoldSamplesRemaining = (int) (currentSampleRate * 0.045);

                float rawMidi = 69.0f + 12.0f * std::log2 (detectedFreq / 440.0f);
                rawMidi += (float) semitoneTranspose;

                if (vocalRangePreset > RangeFollow && manualNote < 0)
                {
                    float minMidi = 48.0f, maxMidi = 72.0f;
                    switch (vocalRangePreset)
                    {
                        case RangeBass:     minMidi = 24.0f; maxMidi = 47.0f; break;
                        case RangeBaritone: minMidi = 45.0f; maxMidi = 68.0f; break;
                        case RangeTenor:    minMidi = 48.0f; maxMidi = 71.0f; break;
                        case RangeAlto:     minMidi = 53.0f; maxMidi = 76.0f; break;
                        case RangeSoprano:  minMidi = 60.0f; maxMidi = 84.0f; break;
                        default: break;
                    }
                    while (rawMidi < minMidi) rawMidi += 12.0f;
                    while (rawMidi > maxMidi) rawMidi -= 12.0f;
                }

                rawMidi = juce::jlimit (12.0f, 115.0f, rawMidi);

                int candidateMidi = (int) std::round (rawMidi);
                float hysteresisWidth = 0.50f + (stickiness * 0.45f);
                if (activeRawMidiNote >= 0 && manualNote < 0
                    && std::abs (rawMidi - (float) activeRawMidiNote) < hysteresisWidth)
                {
                    candidateMidi = activeRawMidiNote;
                }

                int snappedMidi = snapToMaqamScale (candidateMidi);

                // Compute Live Vibrato / Free Pitch cents around target degree (including quarter-tone)
                auto computeTargetCentsForNote = [&] (int noteMidi) -> float
                {
                    int pc = ((noteMidi % 12) + 12) % 12;
                    float baseQtCents = (noteStates[pc] == 2) ? quarterToneCents : 0.0f;
                    if (! vibratoEnabled.load() || manualNote >= 0)
                    {
                        smoothedVibCents = 0.0f;
                        return baseQtCents;
                    }

                    const float vibMaxTable[3] = { 30.0f, 65.0f, 110.0f };
                    const float vibScaleTable[3] = { 0.45f, 0.78f, 1.0f };
                    int vIdx = juce::jlimit (0, 2, vibratoDepthMode.load());

                    float idealMidi = (float) noteMidi + (baseQtCents / 100.0f);
                    float rawDevCents = (rawMidi - idealMidi) * 100.0f;
                    float scaledDev = juce::jlimit (-vibMaxTable[vIdx], vibMaxTable[vIdx], rawDevCents * vibScaleTable[vIdx]);
                    smoothedVibCents = 0.72f * smoothedVibCents + 0.28f * scaledDev;
                    return baseQtCents + smoothedVibCents;
                };

                if (snappedMidi == activeOutputMidiNote)
                {
                    pendingCandidateMidi = -1;
                    pendingCandidateSamples = 0;

                    float totalCents = computeTargetCentsForNote (snappedMidi);
                    int pbValue = computePitchWheelForCents (totalCents);
                    currentPitchBendValue.store (pbValue);
                    currentCentsOffset.store ((int) std::round (totalCents));

                    traceLiveMidi = vibratoEnabled.load() ? ((float) snappedMidi + totalCents / 100.0f) : rawMidi;
                    traceTargetMidi = (float) snappedMidi + (((noteStates[((snappedMidi % 12) + 12) % 12] == 2) ? quarterToneCents : 0.0f) / 100.0f);
                    traceIsValid = true;

                    if (mode == ModeMelodyOnly)
                    {
                        pbThrottleSamples += numSamples;
                        if (pbThrottleSamples >= (int) (currentSampleRate * 0.010))
                        {
                            pbThrottleSamples = 0;
                            emitPitchWheelIfChanged (pbValue, midiMessages, 0, false);
                        }
                    }
                }
                else
                {
                    bool shouldCommit = (manualNote >= 0);
                    if (! shouldCommit)
                    {
                        float lockMs = (mode == ModeChordsOnly) ? chordLockMs : melodyNoteLockMs;
                        int requiredSamples = (int) (currentSampleRate * (lockMs / 1000.0f));

                        if (snappedMidi == pendingCandidateMidi)
                            pendingCandidateSamples += numSamples;
                        else
                        {
                            pendingCandidateMidi = snappedMidi;
                            pendingCandidateSamples = numSamples;
                        }

                        if (pendingCandidateSamples >= requiredSamples)
                            shouldCommit = true;
                    }

                    traceLiveMidi = rawMidi;
                    traceTargetMidi = (activeOutputMidiNote >= 0) ? (float) activeOutputMidiNote : (float) snappedMidi;
                    traceIsValid = true;

                    if (shouldCommit)
                    {
                        pendingCandidateMidi = -1;
                        pendingCandidateSamples = 0;
                        activeRawMidiNote = candidateMidi;
                        smoothedVibCents = 0.0f;

                        int pc = ((snappedMidi % 12) + 12) % 12;
                        float totalCents = computeTargetCentsForNote (snappedMidi);
                        int pbValue = computePitchWheelForCents (totalCents);
                        currentPitchBendValue.store (pbValue);
                        currentCentsOffset.store ((int) std::round (totalCents));

                        juce::uint8 dynVel = (manualNote >= 0) ? 110 : computeDynamicVelocity (rms, gateOpenThresh);
                        int oldNote = activeOutputMidiNote;

                        if (mode == ModeMelodyOnly)
                        {
                            stopActiveChord (midiMessages);
                            int vMode = melodyVoiceMode.load();

                            if (vMode == VoiceMono)
                            {
                                flushPolyAndLegatoOffs (midiMessages);
                                if (oldNote >= 0)
                                    emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, oldNote), midiMessages, 0);

                                int onOffset = (oldNote >= 0 && maxSampleIdx > 0) ? 1 : 0;
                                emitPitchWheelIfChanged (pbValue, midiMessages, onOffset, true);
                                emitMidiEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, dynVel), midiMessages, onOffset);
                            }
                            else if (vMode == VoiceLegato)
                            {
                                flushPolyVoices (midiMessages);
                                if (pendingLegatoOffNote >= 0 && pendingLegatoOffNote != snappedMidi)
                                {
                                    emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, pendingLegatoOffNote), midiMessages, 0);
                                    pendingLegatoOffNote = -1;
                                }

                                emitPitchWheelIfChanged (pbValue, midiMessages, 0, true);
                                emitMidiEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, dynVel), midiMessages, 0);

                                if (oldNote >= 0 && oldNote != snappedMidi)
                                {
                                    pendingLegatoOffNote = oldNote;
                                    legatoOffSamplesRemaining = (int) (currentSampleRate * 0.015);
                                }
                            }
                            else
                            {
                                if (pendingLegatoOffNote >= 0)
                                {
                                    emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, pendingLegatoOffNote), midiMessages, 0);
                                    pendingLegatoOffNote = -1;
                                }

                                if (oldNote >= 0 && oldNote != snappedMidi)
                                {
                                    polyMelodyVoices.push_back ({ oldNote, (int) (currentSampleRate * 0.45) });
                                    if (polyMelodyVoices.size() > 3)
                                    {
                                        emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, polyMelodyVoices.front().note), midiMessages, 0);
                                        polyMelodyVoices.erase (polyMelodyVoices.begin());
                                    }
                                }

                                for (auto it = polyMelodyVoices.begin(); it != polyMelodyVoices.end(); )
                                {
                                    if (it->note == snappedMidi)
                                    {
                                        emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, it->note), midiMessages, 0);
                                        it = polyMelodyVoices.erase (it);
                                    }
                                    else ++it;
                                }

                                int onOffset = (maxSampleIdx > 0) ? 1 : 0;
                                emitPitchWheelIfChanged (pbValue, midiMessages, onOffset, true);
                                emitMidiEvent (juce::MidiMessage::noteOn (pitchMidiChannel, snappedMidi, dynVel), midiMessages, onOffset);
                            }

                            activeOutputMidiNote = snappedMidi;
                            currentDetectedMidiNote.store (snappedMidi);
                        }
                        else if (mode == ModeChordsOnly)
                        {
                            int oldPc = (oldNote >= 0) ? ((oldNote % 12) + 12) % 12 : -1;
                            if (oldPc != pc || activeChordNotes.empty())
                            {
                                flushPolyAndLegatoOffs (midiMessages);
                                if (oldNote >= 0 && activeChordNotes.empty())
                                    emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, oldNote), midiMessages, 0);

                                activeOutputMidiNote = snappedMidi;
                                currentDetectedMidiNote.store (snappedMidi);
                                triggerSmartOrientalChord (snappedMidi, dynVel, midiMessages);
                            }
                        }
                    }
                }
            }
            else
            {
                pendingCandidateMidi = -1;
                pendingCandidateSamples = 0;

                if (activeOutputMidiNote >= 0 || ! activeChordNotes.empty())
                {
                    if (manualNote < 0 && releaseHoldSamplesRemaining > 0)
                    {
                        releaseHoldSamplesRemaining -= numSamples;
                    }
                    else
                    {
                        stopActiveMelodyAndChords (midiMessages);
                    }
                }
            }

            if (sendAutomationCC.load())
            {
                ccThrottleSamples += numSamples;
                if (ccThrottleSamples >= (int) (currentSampleRate * 0.012))
                {
                    ccThrottleSamples = 0;
                    int targetCC = (activeOutputMidiNote >= 0)
                        ? juce::jlimit (0, 127, (int) std::round (vowelEnv * 127.0f))
                        : 0;

                    if (targetCC != lastSentModCC)
                    {
                        int outCh = (mode == ModeChordsOnly) ? chordsMidiChannel : pitchMidiChannel;
                        emitMidiEvent (juce::MidiMessage::controllerEvent (outCh, 1, targetCC), midiMessages, 0);
                        emitMidiEvent (juce::MidiMessage::controllerEvent (outCh, 11, targetCC), midiMessages, 0);
                        lastSentModCC = targetCC;
                    }
                }
            }
        }

        // Push live scrolling pitch analyzer history (~30-40 Hz)
        traceThrottleSamples += numSamples;
        if (traceThrottleSamples >= (int) (currentSampleRate * 0.024))
        {
            traceThrottleSamples = 0;
            pushPitchTracePoint (traceLiveMidi, traceTargetMidi, traceIsValid);
        }

        float* outL = buffer.getWritePointer (0);
        float* outR = (numOutChannels > 1) ? buffer.getWritePointer (1) : outL;

        if (! intellAudioMuted.load() && activeOutputMidiNote >= 0 && mode != ModeTriggersOnly)
        {
            float cents = (float) currentCentsOffset.load();
            double targetFreq = 440.0 * std::pow (2.0, ((double) activeOutputMidiNote - 69.0 + (double) cents / 100.0) / 12.0);

            if (! hadPreviousSynthNote || glideMs <= 1.0f)
                currentSynthFreq = targetFreq;

            hadPreviousSynthNote = true;
            double glideCoeff = (glideMs > 1.0f)
                ? (1.0 - std::exp (-1.0 / (currentSampleRate * ((double) glideMs / 1000.0) * 0.35)))
                : 1.0;

            for (int i = 0; i < numSamples; ++i)
            {
                currentSynthFreq += (targetFreq - currentSynthFreq) * glideCoeff;
                double phaseInc = currentSynthFreq / currentSampleRate;
                float s = renderInternalSynthSample (synthPhase);
                synthPhase += phaseInc;
                if (synthPhase >= 1.0) synthPhase -= 1.0;
                outL[i] += s * 0.22f;
                outR[i] += s * 0.22f;
            }
        }
        else if (activeOutputMidiNote < 0)
        {
            hadPreviousSynthNote = false;
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

    int computePitchWheelForCents (float targetCents) const
    {
        float maxBendCents = (float) pbRangeSemitones * 100.0f;
        int pbValue = 8192 + (int) std::round ((targetCents / maxBendCents) * 8192.0f);
        return juce::jlimit (0, 16383, pbValue);
    }

    void flushPolyVoices (juce::MidiBuffer& midiMessages)
    {
        for (const auto& pv : polyMelodyVoices)
            emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, pv.note), midiMessages, 0);
        polyMelodyVoices.clear();
    }

    void flushPolyAndLegatoOffs (juce::MidiBuffer& midiMessages)
    {
        if (pendingLegatoOffNote >= 0)
        {
            emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, pendingLegatoOffNote), midiMessages, 0);
            pendingLegatoOffNote = -1;
            legatoOffSamplesRemaining = 0;
        }
        flushPolyVoices (midiMessages);
    }

    void stopActiveMelodyAndChords (juce::MidiBuffer& midiMessages)
    {
        flushPolyAndLegatoOffs (midiMessages);
        if (activeOutputMidiNote >= 0)
        {
            emitMidiEvent (juce::MidiMessage::noteOff (pitchMidiChannel, activeOutputMidiNote), midiMessages, 0);
            emitPitchWheelIfChanged (8192, midiMessages, 0, true);
        }
        if (sendAutomationCC.load() && lastSentModCC > 0)
        {
            emitMidiEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, 1, 0), midiMessages, 0);
            emitMidiEvent (juce::MidiMessage::controllerEvent (pitchMidiChannel, 11, 0), midiMessages, 0);
            lastSentModCC = 0;
        }
        stopActiveChord (midiMessages);
        activeOutputMidiNote = -1;
        activeRawMidiNote = -1;
        pendingCandidateMidi = -1;
        pendingCandidateSamples = 0;
        releaseHoldSamplesRemaining = 0;
        smoothedVibCents = 0.0f;
        hadPreviousSynthNote = false;
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

    void fireTriggerPad (int padIdx, juce::uint8 velocity, juce::MidiBuffer& midiMessages, int sampleOffset, int numSamplesInBlock)
    {
        auto& pad = pads[padIdx];
        pad.flashAlpha = 1.0f;

        // Per-pad Level / Velocity Trim scales both outgoing MIDI velocity and internal sample gain
        int trimmedVel = juce::jlimit (1, 127, (int) std::round ((float) velocity * pad.levelTrim));
        lastPadGain[padIdx] = ((float) trimmedVel / 127.0f) * 0.80f;
        if (pad.hasCustomSample) pad.playhead = 0;

        int maxIdx    = juce::jmax (0, numSamplesInBlock - 1);
        int onOffset  = (maxIdx > 0) ? juce::jlimit (0, maxIdx - 1, sampleOffset) : 0;
        int offOffset = (maxIdx > 0) ? juce::jmin (onOffset + 32, maxIdx) : 0;

        emitMidiEvent (juce::MidiMessage::noteOn  (triggersMidiChannel, pad.midiNote, (juce::uint8) trimmedVel), midiMessages, onOffset);
        emitMidiEvent (juce::MidiMessage::noteOff (triggersMidiChannel, pad.midiNote), midiMessages, offOffset);
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
    const juce::String getName() const override { return "VoxNI"; }
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
        auto state = buildStateValueTree();
        std::unique_ptr<juce::XmlElement> xml (state.createXml());
        copyXmlToBinary (*xml, destData);
    }

    void setStateInformation (const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
        if (xmlState != nullptr)
            applyStateValueTree (juce::ValueTree::fromXml (*xmlState));
    }

    static constexpr int kPitchTraceLen = 96;
    std::array<float, kPitchTraceLen> pitchTraceMidi {};
    std::array<float, kPitchTraceLen> pitchTraceTargetMidi {};
    std::array<bool,  kPitchTraceLen> pitchTraceValid {};
    std::atomic<int> pitchTraceWriteIdx { 0 };

    std::array<TriggerPadData, 8> pads;
    std::array<int, 12> noteStates { 1, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 2 };
    std::array<int, 7> degreePitchClasses { 0, 2, 4, 5, 7, 9, 11 };
    std::array<std::vector<ChordVoicingOption>, 7> degreeChordOptions;
    std::array<int, 7> selectedDegreeVoicing { 0, 0, 0, 0, 0, 0, 0 };

    int currentPresetIndex = 0;
    juce::String currentPresetName { "1. Oriental Wind / Nay" };

    int rootNote = 0;
    int currentMaqamIndex = 2;
    float quarterToneCents = -50.0f;
    float stickiness = 0.65f;

    std::atomic<int> melodyVoiceMode { VoiceMono };
    float glideMs = 0.0f;
    float melodyNoteLockMs = 28.0f;
    float chordLockMs = 75.0f;
    float triggerRetrigLockMs = 85.0f;
    std::atomic<bool> pendingGlideCCUpdate { true };

    std::atomic<bool> vibratoEnabled { true };
    std::atomic<int>  vibratoDepthMode { 1 };   // 0=Soft, 1=Med, 2=Deep
    std::atomic<bool> sendAutomationCC { true };
    std::atomic<int>  ccSensitivityMode { 1 };  // 0=Soft, 1=Med, 2=High

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
    juce::String autoRootStatusText { "AUTO ROOT" };

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

private:
    static constexpr int kYinWindowSize = 2048;
    std::vector<float> yinBuffer;
    int yinWritePos = 0;
    int samplesSinceLastYin = 0;
    float lastDetectedFreq = -1.0f;
    float lastPitchConfidence = 0.0f;

    double currentSampleRate = 44100.0;
    double synthPhase = 0.0;
    double currentSynthFreq = 261.63;
    bool hadPreviousSynthNote = false;

    float lpFilterState = 0.0f;
    float hpCondenserState = 0.0f;
    float prevSampleRaw = 0.0f;
    float prevSamplePreEmp = 0.0f;

    float pitchHpState = 0.0f;
    float pitchPrevRaw = 0.0f;
    float pitchLp1 = 0.0f;
    float pitchLp2 = 0.0f;
    float pitchLpAlpha = 0.145f;
    float pitchHpCoeff = 0.989f;

    bool globalGateIsOpen = false;
    float calibPeakRms = 0.0f;
    int calibSamplesRemaining = 0;

    juce::SpinLock autoKeyLock;
    std::array<float, 12> pitchClassWeights {};
    int lastDetectedPitchClass = -1;
    int totalDetectedVocalFrames = 0;
    int autoKeySamplesElapsed = 0;

    int activeRawMidiNote = -1, activeOutputMidiNote = -1;
    int pendingCandidateMidi = -1;
    int pendingCandidateSamples = 0;
    int releaseHoldSamplesRemaining = 0;

    int pendingLegatoOffNote = -1;
    int legatoOffSamplesRemaining = 0;
    std::vector<PolyVoice> polyMelodyVoices;

    float smoothedVibCents = 0.0f;
    int lastSentPitchWheel = -1;
    int lastSentModCC = -1;
    int ccThrottleSamples = 0;
    int pbThrottleSamples = 0;
    int traceThrottleSamples = 0;

    float prevTrigLevel = 0.0f;
    int triggerCooldownSamples = 0;
    std::array<float, 8> lastPadGain { 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f, 0.65f };

    std::vector<int> activeChordNotes;
    juce::AudioFormatManager formatManager;
    juce::CriticalSection sampleLock;

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
        logoImage = juce::ImageCache::getFromMemory (BinaryData::voxni_logo_png, BinaryData::voxni_logo_pngSize);
        if (logoImage.isValid() && logoImage.getHeight() > (int) (logoImage.getWidth() * 0.65f))
        {
            int cropX = (int) (logoImage.getWidth()  * 0.12f);
            int cropW = (int) (logoImage.getWidth()  * 0.76f);
            int cropY = (int) (logoImage.getHeight() * 0.35f);
            int cropH = (int) (logoImage.getHeight() * 0.30f);
            logoImage = logoImage.getClippedImage ({ cropX, cropY, cropW, cropH });
        }

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

    void openSavePresetDialog()
    {
        fileChooser = std::make_unique<juce::FileChooser> (
            "Save VoxNI Preset", proc.getUserPresetsDirectory().getChildFile ("MyPreset.voxni"), "*.voxni");
        fileChooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
            [this] (const juce::FileChooser& fc)
            {
                auto file = fc.getResult();
                if (file != juce::File{})
                {
                    if (! file.hasFileExtension (".voxni"))
                        file = file.withFileExtension (".voxni");
                    proc.savePresetToFile (file);
                    repaint();
                }
            });
    }

    void showPresetsMenu()
    {
        juce::PopupMenu menu;
        menu.addSectionHeader ("Factory Presets");
        for (int i = 0; i < (int) kFactoryPresetNames.size(); ++i)
            menu.addItem (1 + i, kFactoryPresetNames[i], true, proc.currentPresetIndex == i);

        auto userFiles = proc.getUserPresetFiles();
        if (! userFiles.isEmpty())
        {
            menu.addSeparator();
            menu.addSectionHeader ("User Presets");
            for (int i = 0; i < userFiles.size(); ++i)
                menu.addItem (100 + i, userFiles[i].getFileNameWithoutExtension(), true,
                              proc.currentPresetIndex < 0 && proc.currentPresetName == userFiles[i].getFileNameWithoutExtension());
        }

        menu.addSeparator();
        menu.addItem (900, "Save Current Settings as Preset...");

        menu.showMenuAsync (juce::PopupMenu::Options().withTargetScreenArea (localAreaToGlobal (presetNameRect.toNearestInt())),
            [this, userFiles] (int result)
            {
                if (result >= 1 && result <= (int) kFactoryPresetNames.size())
                {
                    proc.applyFactoryPreset (result - 1);
                    repaint();
                }
                else if (result >= 100 && result < 100 + userFiles.size())
                {
                    proc.loadPresetFromFile (userFiles[result - 100]);
                    repaint();
                }
                else if (result == 900)
                {
                    openSavePresetDialog();
                }
            });
    }

    void paint (juce::Graphics& g) override
    {
        const juce::Colour bgDark (0xff1b1b1f), cardDark (0xff25252b), cyberGold (0xffe5b84b);
        const juce::Colour neonPink (0xfff03e76), neonGreen (0xff2ecc71), textBright (0xfff4f0e8), textMuted (0xff8e8d96);

        g.fillAll (bgDark);
        g.setColour (juce::Colour (0xff171717));
        g.fillRect (0, 0, getWidth(), 42);

        // Logo (Clickable -> Waves-style About Popup)
        logoClickRect = { 8.0f, 4.0f, 88.0f, 34.0f };
        if (logoImage.isValid())
        {
            g.drawImageWithin (logoImage, 8, 5, 88, 32, juce::RectanglePlacement::centred);
        }
        else
        {
            g.setFont (juce::FontOptions (17.0f).withStyle ("Bold"));
            g.setColour (cyberGold);
            g.drawText ("VoxNI", logoClickRect, juce::Justification::centredLeft);
        }

        // 3 Exclusive Modes
        const int activeMode = proc.exclusiveMode.load();
        const char* modeNames[3] = { "1. MELODY", "2. TRIGGERS", "3. CHORDS" };
        const float modeX[3] = { 102.0f, 188.0f, 280.0f };
        const float modeW[3] = { 82.0f,  88.0f,  82.0f };
        for (int m = 0; m < 3; ++m)
        {
            exclusiveModeRects[m] = { modeX[m], 8.0f, modeW[m], 26.0f };
            bool isCurr = (activeMode == m);
            g.setColour (isCurr ? (m == 1 ? neonPink : cyberGold) : juce::Colour (0xff23232a));
            g.fillRoundedRectangle (exclusiveModeRects[m], 5.0f);
            g.setColour (isCurr ? juce::Colours::black : textBright);
            g.setFont (juce::FontOptions (10.0f).withStyle ("Bold"));
            g.drawText (modeNames[m], exclusiveModeRects[m], juce::Justification::centred);
        }

        // Presets Bar (< [Preset Name] > [SAVE])
        presetPrevRect = { 368.0f, 8.0f, 20.0f, 26.0f };
        presetNameRect = { 390.0f, 8.0f, 136.0f, 26.0f };
        presetNextRect = { 528.0f, 8.0f, 20.0f, 26.0f };
        presetSaveRect = { 552.0f, 8.0f, 40.0f, 26.0f };

        g.setColour (juce::Colour (0xff26262f));
        g.fillRoundedRectangle (presetPrevRect, 4.0f);
        g.fillRoundedRectangle (presetNameRect, 4.0f);
        g.fillRoundedRectangle (presetNextRect, 4.0f);
        g.setColour (cyberGold.withAlpha (0.45f));
        g.drawRoundedRectangle (presetNameRect, 4.0f, 1.0f);

        g.setColour (cyberGold);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("<", presetPrevRect, juce::Justification::centred);
        g.drawText (">", presetNextRect, juce::Justification::centred);
        g.setColour (textBright);
        g.setFont (juce::FontOptions (10.0f).withStyle ("Bold"));
        g.drawFittedText (proc.currentPresetName, presetNameRect.reduced (4.0f, 1.0f).toNearestInt(), juce::Justification::centred, 1);

        g.setColour (juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (presetSaveRect, 4.0f);
        g.setColour (neonGreen);
        g.setFont (juce::FontOptions (9.5f).withStyle ("Bold"));
        g.drawText ("SAVE", presetSaveRect, juce::Justification::centred);

        // MIDI Capture Controls
        bool isRec = proc.isCapturingMidi.load();
        int noteCnt = proc.capturedNoteCount.load();

        midiRecButtonRect = { 598.0f, 8.0f, 82.0f, 26.0f };
        g.setColour (isRec ? neonPink : juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (midiRecButtonRect, 5.0f);
        g.setColour (isRec ? textBright : neonPink);
        g.fillEllipse (midiRecButtonRect.getX() + 7.0f, midiRecButtonRect.getY() + 9.0f, 7.0f, 7.0f);
        g.setColour (textBright);
        g.setFont (juce::FontOptions (10.0f).withStyle ("Bold"));
        g.drawText (isRec ? ("STOP " + juce::String (noteCnt)) : "REC MIDI",
                    midiRecButtonRect.withTrimmedLeft (13.0f), juce::Justification::centred);

        quantizeCycleRect = { 684.0f, 8.0f, 76.0f, 26.0f };
        g.setColour (proc.quantizeMode > 0 ? juce::Colour (0xff2d2a1e) : juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (quantizeCycleRect, 5.0f);
        g.setColour (cyberGold);
        const char* qLabels[3] = { "Q: OFF", "Q: 1/8", "Q: 1/16" };
        g.drawText (qLabels[proc.quantizeMode % 3], quantizeCycleRect, juce::Justification::centred);

        midiDragButtonRect = { 764.0f, 8.0f, 134.0f, 26.0f };
        g.setColour (noteCnt > 0 ? neonGreen : cyberGold);
        g.fillRoundedRectangle (midiDragButtonRect, 5.0f);
        g.setColour (juce::Colours::black);
        g.drawText ("DRAG MIDI (" + juce::String (noteCnt) + ")", midiDragButtonRect, juce::Justification::centred);

        midiClearButtonRect = { 902.0f, 8.0f, 48.0f, 26.0f };
        g.setColour (juce::Colour (0xff2a2a32));
        g.fillRoundedRectangle (midiClearButtonRect, 5.0f);
        g.setColour (textBright);
        g.drawText ("CLR", midiClearButtonRect, juce::Justification::centred);

        // 5 Tabs
        const char* tabTitles[5] = { "Play", "Train (Beatbox)", "Key (Root/Maqam)", "Chords (Matrix)", "Assign & Mic" };
        for (int i = 0; i < 5; ++i)
        {
            tabRects[i] = { 130.0f + i * 144.0f, 46.0f, 138.0f, 30.0f };
            bool isCurr = (currentTab == i);
            g.setColour (isCurr ? cardDark : juce::Colour (0xff19191e));
            g.fillRoundedRectangle (tabRects[i], 6.0f);
            g.setColour (isCurr ? cyberGold : textMuted);
            g.setFont (juce::FontOptions (11.5f).withStyle ("Bold"));
            g.drawText (tabTitles[i], tabRects[i], juce::Justification::centred);
        }

        // Main Card
        juce::Rectangle<float> mainCard (28.0f, 74.0f, 920.0f, 336.0f);
        g.setColour (cardDark);
        g.fillRoundedRectangle (mainCard, 10.0f);
        g.setColour (cyberGold.withAlpha (0.28f));
        g.drawRoundedRectangle (mainCard, 10.0f, 1.2f);

        // Left Global Gate LED Meter
        globalGateMeterRect = { 7.0f, 84.0f, 14.0f, 316.0f };
        float rmsNorm = juce::jlimit (0.0f, 1.0f, proc.currentInputRms.load() * 14.0f);
        for (int d = 0; d < 28; ++d)
        {
            float dy = 394.0f - d * 11.0f;
            bool lit = ((float) d / 28.0f) <= rmsNorm;
            g.setColour (lit ? (d > 18 ? neonGreen : cyberGold) : textMuted.withAlpha (0.25f));
            g.fillEllipse (11.0f, dy, 4.5f, 4.5f);
        }
        float gateY = 396.0f - proc.globalGateLevel * 308.0f;
        g.setColour (neonPink);
        g.fillRect (5.0f, gateY - 1.5f, 18.0f, 3.0f);

        if      (currentTab == 0) drawPlayTab   (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 1) drawTrainTab  (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 2) drawKeyTab    (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 3) drawChordsTab (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
        else if (currentTab == 4) drawAssignTab (g, cyberGold, neonPink, neonGreen, textBright, textMuted);

        drawBottomSection (g, cyberGold, neonPink, neonGreen, textBright, textMuted);

        // Draw Floating Arabic Hover Tooltip (when About Popup is closed)
        if (! showAboutPopup && currentTooltipText.isNotEmpty())
            drawArabicTooltip (g, cyberGold, textBright);

        // Draw Waves-Style About Popup Modal on top of everything
        if (showAboutPopup)
            drawAboutPopupModal (g, cyberGold, neonPink, neonGreen, textBright, textMuted);
    }

    void drawLivePitchAnalyzer (juce::Graphics& g, juce::Rectangle<float> area,
                                juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        pitchAnalyzerRect = area;
        g.setColour (juce::Colour (0xff15151a));
        g.fillRoundedRectangle (area, 7.0f);
        g.setColour (gold.withAlpha (0.35f));
        g.drawRoundedRectangle (area, 7.0f, 1.1f);

        // Gridlines
        g.setColour (muted.withAlpha (0.14f));
        for (int row = 1; row <= 3; ++row)
        {
            float gy = area.getY() + (area.getHeight() * (float) row / 4.0f);
            g.drawHorizontalLine ((int) gy, area.getX() + 4.0f, area.getRight() - 4.0f);
        }

        // Header & Live Cents Badge
        g.setColour (gold);
        g.setFont (juce::FontOptions (9.5f).withStyle ("Bold"));
        g.drawText ("LIVE PITCH ANALYZER", (int) area.getX() + 8, (int) area.getY() + 3, 130, 14, juce::Justification::centredLeft);

        int activeMidi = proc.currentDetectedMidiNote.load();
        int centsOff = proc.currentCentsOffset.load();
        juce::String badge = (activeMidi >= 0)
            ? (juce::String (kNoteNames[((activeMidi % 12) + 12) % 12]) + juce::String (activeMidi / 12 - 1)
               + " (" + (centsOff >= 0 ? "+" : "") + juce::String (centsOff) + "c)")
            : "IDLE";
        g.setColour (activeMidi >= 0 ? green : muted);
        g.drawText (badge, (int) area.getRight() - 108, (int) area.getY() + 3, 100, 14, juce::Justification::centredRight);

        // Determine dynamic center MIDI for graph vertical framing
        int writeIdx = proc.pitchTraceWriteIdx.load();
        float centerMidi = 60.0f;
        for (int i = 1; i <= VoxN2AudioProcessor::kPitchTraceLen; ++i)
        {
            int idx = (writeIdx - i + VoxN2AudioProcessor::kPitchTraceLen) % VoxN2AudioProcessor::kPitchTraceLen;
            if (proc.pitchTraceValid[idx]) { centerMidi = proc.pitchTraceTargetMidi[idx]; break; }
        }

        const float plotX = area.getX() + 6.0f;
        const float plotW = area.getWidth() - 12.0f;
        const float plotY = area.getY() + 20.0f;
        const float plotH = area.getHeight() - 26.0f;
        const float semiSpan = 8.0f; // +/- 4 semitones vertical range

        auto midiToY = [&] (float m) -> float
        {
            float norm = 0.5f - ((m - centerMidi) / semiSpan);
            return plotY + juce::jlimit (0.04f, 0.96f, norm) * plotH;
        };

        juce::Path targetPath, livePath;
        bool inTarget = false, inLive = false;

        for (int s = 0; s < VoxN2AudioProcessor::kPitchTraceLen; ++s)
        {
            int idx = (writeIdx + s) % VoxN2AudioProcessor::kPitchTraceLen;
            float px = plotX + ((float) s / (float) (VoxN2AudioProcessor::kPitchTraceLen - 1)) * plotW;

            if (proc.pitchTraceValid[idx])
            {
                float ty = midiToY (proc.pitchTraceTargetMidi[idx]);
                float ly = midiToY (proc.pitchTraceMidi[idx]);

                if (! inTarget) { targetPath.startNewSubPath (px, ty); inTarget = true; }
                else targetPath.lineTo (px, ty);

                if (! inLive) { livePath.startNewSubPath (px, ly); inLive = true; }
                else livePath.lineTo (px, ly);
            }
            else
            {
                inTarget = false;
                inLive = false;
            }
        }

        g.setColour (gold.withAlpha (0.42f));
        g.strokePath (targetPath, juce::PathStrokeType (1.5f));
        g.setColour (proc.vibratoEnabled.load() ? pink : green);
        g.strokePath (livePath, juce::PathStrokeType (2.4f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Glowing head dot on active vocal pitch
        int lastIdx = (writeIdx - 1 + VoxN2AudioProcessor::kPitchTraceLen) % VoxN2AudioProcessor::kPitchTraceLen;
        if (proc.pitchTraceValid[lastIdx])
        {
            float hy = midiToY (proc.pitchTraceMidi[lastIdx]);
            g.setColour (bright);
            g.fillEllipse (plotX + plotW - 4.0f, hy - 3.5f, 7.0f, 7.0f);
        }
    }

    void drawPlayTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::FontOptions (11.0f).withStyle ("Bold"));
        g.drawText ("Trig Sens: " + juce::String ((int) (proc.triggerSensitivity * 100.0f)) + "%", 44, 80, 95, 20, juce::Justification::centredLeft);

        playTrigSensRect = { 140.0f, 84.0f, 170.0f, 12.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (playTrigSensRect, 6.0f);
        g.setColour (green);
        g.fillRoundedRectangle (playTrigSensRect.getX(), playTrigSensRect.getY(),
                                playTrigSensRect.getWidth() * proc.triggerSensitivity, 12.0f, 6.0f);

        for (int i = 0; i < 8; ++i)
        {
            int row = i / 3, col = i % 3;
            juce::Rectangle<float> r (44.0f + col * 92.0f, 108.0f + row * 96.0f, 84.0f, 88.0f);
            padBounds[i] = r;
            auto& pad = proc.pads[i];

            g.setColour (juce::Colour (0xff1d1d22).interpolatedWith (pink, pad.flashAlpha * 0.55f));
            g.fillRoundedRectangle (r, 7.0f);
            g.setColour (pad.hasCustomSample ? gold : muted.withAlpha (0.4f));
            g.drawRoundedRectangle (r, 7.0f, 1.4f);

            padLoadRects[i] = { r.getX() + 4.0f, r.getY() + 4.0f, 34.0f, 15.0f };
            g.setColour (gold.withAlpha (0.25f));
            g.fillRoundedRectangle (padLoadRects[i], 3.0f);
            g.setColour (gold);
            g.setFont (juce::FontOptions (8.5f));
            g.drawText ("LOAD", padLoadRects[i], juce::Justification::centred);

            if (pad.isVoiceTrained)
            {
                g.setColour (green);
                g.drawText ("MIC OK", (int) r.getRight() - 40, (int) r.getY() + 4, 36, 15, juce::Justification::centredRight);
            }

            g.setColour (bright);
            g.setFont (juce::FontOptions (11.0f).withStyle ("Bold"));
            g.drawText (pad.name, (int) r.getX() + 5, (int) r.getY() + 26, 74, 16, juce::Justification::centredLeft);

            g.setColour (pad.hasCustomSample ? gold : muted);
            g.setFont (juce::FontOptions (9.0f));
            g.drawText (pad.hasCustomSample ? pad.customFileName : "[Empty Pad]",
                        (int) r.getX() + 5, (int) r.getY() + 43, 74, 15, juce::Justification::centredLeft);

            // Per-Pad Level / Velocity Trim Bar
            padLevelRects[i] = { r.getX() + 5.0f, r.getBottom() - 18.0f, r.getWidth() - 10.0f, 13.0f };
            g.setColour (juce::Colour (0xff121216));
            g.fillRoundedRectangle (padLevelRects[i], 3.5f);
            g.setColour (gold.withAlpha (0.85f));
            g.fillRoundedRectangle (padLevelRects[i].getX(), padLevelRects[i].getY(),
                                    padLevelRects[i].getWidth() * pad.levelTrim, padLevelRects[i].getHeight(), 3.5f);
            g.setColour (juce::Colours::black);
            g.setFont (juce::FontOptions (8.5f).withStyle ("Bold"));
            g.drawText ("LVL " + juce::String ((int) (pad.levelTrim * 100.0f)) + "%", padLevelRects[i], juce::Justification::centred);
        }

        // Center Top: Live Scrolling Pitch Analyzer (Auto-Tune Style)
        drawLivePitchAnalyzer (g, { 330.0f, 82.0f, 312.0f, 102.0f }, gold, pink, green, bright, muted);

        // Center Bottom: Pitch Radar Wheel
        drawPitchRadarWheel (g, 486.0f, 294.0f, 98.0f, gold, pink, bright);

        // Right Column: Melody Controls + Velocity Curve
        g.setColour (muted);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("Stickiness: " + juce::String ((int) (proc.stickiness * 100.0f)) + "%", 655, 80, 96, 16, juce::Justification::centredLeft);
        stickinessSliderRect = { 754.0f, 83.0f, 171.0f, 10.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (stickinessSliderRect, 5.0f);
        g.setColour (pink);
        g.fillRoundedRectangle (stickinessSliderRect.getX(), stickinessSliderRect.getY(),
                                stickinessSliderRect.getWidth() * proc.stickiness, 10.0f, 5.0f);

        const int vMode = proc.melodyVoiceMode.load();
        const char* vNames[3] = { "MONO", "LEGATO", "POLY" };
        for (int v = 0; v < 3; ++v)
        {
            voiceModeRects[v] = { 655.0f + v * 92.0f, 102.0f, 86.0f, 24.0f };
            bool isSel = (vMode == v);
            g.setColour (isSel ? (v == 1 ? pink : (v == 2 ? green : gold)) : juce::Colour (0xff1c1c23));
            g.fillRoundedRectangle (voiceModeRects[v], 5.0f);
            g.setColour (isSel ? juce::Colours::black : bright);
            g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
            g.drawText (vNames[v], voiceModeRects[v], juce::Justification::centred);
        }

        g.setColour (gold);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("Glide: " + juce::String ((int) proc.glideMs) + " ms", 655, 132, 95, 16, juce::Justification::centredLeft);
        glideSliderRect = { 754.0f, 135.0f, 171.0f, 10.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (glideSliderRect, 5.0f);
        g.setColour (gold);
        g.fillRoundedRectangle (glideSliderRect.getX(), glideSliderRect.getY(),
                                glideSliderRect.getWidth() * (proc.glideMs / 300.0f), 10.0f, 5.0f);

        g.setColour (green);
        g.drawText ("Note Lock: " + juce::String ((int) proc.melodyNoteLockMs) + " ms", 655, 154, 96, 16, juce::Justification::centredLeft);
        melodyLockSliderRect = { 754.0f, 157.0f, 171.0f, 10.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (melodyLockSliderRect, 5.0f);
        float lockNorm = juce::jlimit (0.0f, 1.0f, (proc.melodyNoteLockMs - 10.0f) / 80.0f);
        g.setColour (green);
        g.fillRoundedRectangle (melodyLockSliderRect.getX(), melodyLockSliderRect.getY(),
                                melodyLockSliderRect.getWidth() * lockNorm, 10.0f, 5.0f);

        g.setColour (gold);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("DYNAMIC VELOCITY CURVE (DRAG):", 655, 176, 270, 16, juce::Justification::centredLeft);

        velocityCurveBoxRect = { 655.0f, 194.0f, 270.0f, 118.0f };
        g.setColour (juce::Colour (0xff18181d));
        g.fillRoundedRectangle (velocityCurveBoxRect, 7.0f);
        g.setColour (gold.withAlpha (0.4f));
        g.drawRoundedRectangle (velocityCurveBoxRect, 7.0f, 1.2f);

        juce::Path curvePath;
        float expVal = std::pow (2.2f, -proc.velocityCurve);
        for (int step = 0; step <= 30; ++step)
        {
            float t = (float) step / 30.0f;
            float yVal = proc.fixedVelocity127 ? 1.0f : std::pow (t, expVal);
            float px = velocityCurveBoxRect.getX() + 10.0f + t * (velocityCurveBoxRect.getWidth() - 20.0f);
            float py = velocityCurveBoxRect.getBottom() - 10.0f - yVal * (velocityCurveBoxRect.getHeight() - 20.0f);
            if (step == 0) curvePath.startNewSubPath (px, py);
            else curvePath.lineTo (px, py);
        }
        g.setColour (proc.fixedVelocity127 ? pink : green);
        g.strokePath (curvePath, juce::PathStrokeType (2.6f));

        fixedVelToggleRect = { 655.0f, 320.0f, 270.0f, 28.0f };
        g.setColour (proc.fixedVelocity127 ? pink : juce::Colour (0xff2c2c35));
        g.fillRoundedRectangle (fixedVelToggleRect, 6.0f);
        g.setColour (bright);
        g.drawText (proc.fixedVelocity127 ? "VELOCITY: FIXED 127" : "VELOCITY: DYNAMIC (DRAG CURVE)",
                    fixedVelToggleRect, juce::Justification::centred);

        playCalibBtnRect = { 655.0f, 356.0f, 270.0f, 30.0f };
        bool cal = proc.isCalibratingRoom.load();
        g.setColour (cal ? pink : juce::Colour (0xff1f2d28));
        g.fillRoundedRectangle (playCalibBtnRect, 6.0f);
        g.setColour (green);
        g.drawRoundedRectangle (playCalibBtnRect, 6.0f, 1.1f);
        g.setColour (bright);
        g.drawText (cal ? "CALIBRATING ROOM..." : "AUTO-CALIBRATE ROOM GATE",
                    playCalibBtnRect, juce::Justification::centred);
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
                g.setFont (juce::FontOptions (10.0f).withStyle ("Bold"));
                g.drawText (juce::String (kNoteNames[i]) + (state == 2 ? " 1/2b" : ""), (int) tx - 22, (int) ty - 9, 44, 18, juce::Justification::centred);
            }
        }

        float innerR = radius * 0.42f;
        g.setColour (juce::Colour (0xff19191f));
        g.fillEllipse (cx - innerR, cy - innerR, innerR * 2.0f, innerR * 2.0f);
        g.setColour (gold);
        g.drawEllipse (cx - innerR, cy - innerR, innerR * 2.0f, innerR * 2.0f, 1.8f);

        juce::String centerNote = (activeMidi >= 0) ? (juce::String (kNoteNames[activePc]) + juce::String (activeMidi / 12 - 1)) : "READY";
        g.setColour (bright);
        g.setFont (juce::FontOptions (13.5f).withStyle ("Bold"));
        g.drawText (centerNote, (int) cx - 36, (int) cy - 11, 72, 22, juce::Justification::centred);
    }

    void drawTrainTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::FontOptions (12.5f).withStyle ("Bold"));
        g.drawText ("BEATBOX & HI-HAT TRAINER (PER-PAD LEVEL & RETRIG LOCK):", 48, 82, 760, 20, juce::Justification::centredLeft);

        for (int i = 0; i < 8; ++i)
        {
            juce::Rectangle<float> r (48.0f + (i % 2) * 146.0f, 110.0f + (i / 2) * 66.0f, 138.0f, 56.0f);
            padBounds[i] = r;
            bool sel = (selectedTrainPad == i);
            g.setColour (sel ? gold.withAlpha (0.22f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (r, 6.0f);
            g.setColour (sel ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (r, 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::FontOptions (11.0f).withStyle ("Bold"));
            g.drawText (proc.pads[i].name + (proc.pads[i].isVoiceTrained ? " [OK]" : ""), r.reduced (7.0f), juce::Justification::topLeft);

            g.setColour (green);
            g.setFont (juce::FontOptions (9.5f).withStyle ("Bold"));
            g.drawText (juce::String ((int) (proc.pads[i].levelTrim * 100.0f)) + "%", r.reduced (7.0f), juce::Justification::topRight);

            g.setColour (proc.pads[i].hasCustomSample ? gold : muted);
            g.setFont (juce::FontOptions (9.5f));
            g.drawText (proc.pads[i].hasCustomSample ? proc.pads[i].customFileName : "[Empty Pad]", r.reduced (7.0f), juce::Justification::bottomLeft);
        }

        auto& currPad = proc.pads[selectedTrainPad];
        bool isListeningNow = (proc.trainingPadIndex.load() == selectedTrainPad);
        float cx = 465.0f, cy = 225.0f;

        g.setColour (juce::Colour (0xff191920));
        g.fillEllipse (cx - 78.0f, cy - 78.0f, 156.0f, 156.0f);
        g.setColour (isListeningNow ? pink : (currPad.isVoiceTrained ? green : gold));
        g.drawEllipse (cx - 78.0f, cy - 78.0f, 156.0f, 156.0f, 3.2f);

        g.setColour (bright);
        g.setFont (juce::FontOptions (13.5f).withStyle ("Bold"));
        juce::String statusText = isListeningNow
            ? ("MAKE SOUND!\nHit " + juce::String (currPad.trainedHits) + " / 3")
            : (currPad.isVoiceTrained ? "TRAINED!\n(Ready)" : "NOT TRAINED\nClick Train ->");
        g.drawFittedText (statusText, (int) cx - 65, (int) cy - 28, 130, 56, juce::Justification::centred, 2);

        g.setColour (muted);
        g.setFont (juce::FontOptions (11.0f));
        g.drawText ("Live Mic Peak:", 370, 322, 190, 18, juce::Justification::centred);
        juce::Rectangle<float> liveMeter (370.0f, 344.0f, 190.0f, 14.0f);
        g.setColour (juce::Colour (0xff141418));
        g.fillRoundedRectangle (liveMeter, 6.0f);
        float levelNorm = juce::jlimit (0.0f, 1.0f, proc.currentInputRms.load() * 16.0f);
        g.setColour (green);
        g.fillRoundedRectangle (liveMeter.getX(), liveMeter.getY(), liveMeter.getWidth() * levelNorm, 14.0f, 6.0f);

        trainVoiceBtnRect     = { 590.0f, 102.0f, 330.0f, 32.0f };
        trainLoadButtonRect   = { 590.0f, 140.0f, 160.0f, 28.0f };
        trainPlayButtonRect   = { 760.0f, 140.0f, 160.0f, 28.0f };
        trainRevertButtonRect = { 590.0f, 174.0f, 330.0f, 24.0f };

        g.setColour (isListeningNow ? green : pink);
        g.fillRoundedRectangle (trainVoiceBtnRect, 6.0f);
        g.setColour (bright);
        g.setFont (juce::FontOptions (11.5f).withStyle ("Bold"));
        g.drawText (isListeningNow ? "LISTENING... MAKE SOUND 3 TIMES!" : "TRAIN VOICE (MAKE SOUND 3X)", trainVoiceBtnRect, juce::Justification::centred);

        g.setColour (gold);
        g.fillRoundedRectangle (trainLoadButtonRect, 6.0f);
        g.setColour (juce::Colours::black);
        g.drawText ("LOAD WAV / MP3", trainLoadButtonRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff32323b));
        g.fillRoundedRectangle (trainPlayButtonRect, 6.0f);
        g.setColour (bright);
        g.drawText ("TEST PAD SOUND", trainPlayButtonRect, juce::Justification::centred);

        g.setColour (juce::Colour (0xff292226));
        g.fillRoundedRectangle (trainRevertButtonRect, 6.0f);
        g.setColour (bright);
        g.setFont (juce::FontOptions (10.5f));
        g.drawText ("CLEAR PAD SAMPLE & TRAINING", trainRevertButtonRect, juce::Justification::centred);

        g.setColour (gold);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("1. Train Mic Sensitivity: " + juce::String ((int) (proc.trainSensitivity * 100.0f)) + "%", 590, 206, 260, 15, juce::Justification::centredLeft);
        trainSensSliderRect = { 590.0f, 222.0f, 330.0f, 11.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (trainSensSliderRect, 5.5f);
        g.setColour (pink);
        g.fillRoundedRectangle (trainSensSliderRect.getX(), trainSensSliderRect.getY(),
                                trainSensSliderRect.getWidth() * proc.trainSensitivity, 11.0f, 5.5f);

        g.setColour (gold);
        g.drawText ("2. Hi-Hat / Soft Boost: " + juce::String ((int) (proc.hiHatBoost * 100.0f)) + "%", 590, 240, 260, 15, juce::Justification::centredLeft);
        hiHatBoostSliderRect = { 590.0f, 256.0f, 330.0f, 11.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (hiHatBoostSliderRect, 5.5f);
        g.setColour (green);
        g.fillRoundedRectangle (hiHatBoostSliderRect.getX(), hiHatBoostSliderRect.getY(),
                                hiHatBoostSliderRect.getWidth() * proc.hiHatBoost, 11.0f, 5.5f);

        g.setColour (gold);
        g.drawText ("3. Pad Sensitivity (" + currPad.name + "): " + juce::String ((int) (currPad.padSensitivity * 100.0f)) + "%", 590, 274, 290, 15, juce::Justification::centredLeft);
        padSensSliderRect = { 590.0f, 290.0f, 330.0f, 11.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (padSensSliderRect, 5.5f);
        g.setColour (gold);
        g.fillRoundedRectangle (padSensSliderRect.getX(), padSensSliderRect.getY(),
                                padSensSliderRect.getWidth() * currPad.padSensitivity, 11.0f, 5.5f);

        g.setColour (green);
        g.drawText ("4. Pad MIDI Velocity / Level (" + currPad.name + "): " + juce::String ((int) (currPad.levelTrim * 100.0f)) + "%", 590, 308, 310, 15, juce::Justification::centredLeft);
        trainPadLevelSliderRect = { 590.0f, 324.0f, 330.0f, 11.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (trainPadLevelSliderRect, 5.5f);
        g.setColour (green);
        g.fillRoundedRectangle (trainPadLevelSliderRect.getX(), trainPadLevelSliderRect.getY(),
                                trainPadLevelSliderRect.getWidth() * currPad.levelTrim, 11.0f, 5.5f);

        g.setColour (pink);
        g.drawText ("5. Retrig Lock (Anti-Double Hit): " + juce::String ((int) proc.triggerRetrigLockMs) + " ms", 590, 342, 300, 15, juce::Justification::centredLeft);
        retrigLockSliderRect = { 590.0f, 358.0f, 330.0f, 11.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (retrigLockSliderRect, 5.5f);
        float rtNorm = juce::jlimit (0.0f, 1.0f, (proc.triggerRetrigLockMs - 40.0f) / 160.0f);
        g.setColour (pink);
        g.fillRoundedRectangle (retrigLockSliderRect.getX(), retrigLockSliderRect.getY(),
                                retrigLockSliderRect.getWidth() * rtNorm, 11.0f, 5.5f);
    }

    void drawKeyTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        drawPitchRadarWheel (g, 200.0f, 212.0f, 108.0f, gold, pink, bright);

        bool detecting = proc.isDetectingKey.load();
        keyTabAutoDetectRect = { 58.0f, 334.0f, 284.0f, 36.0f };
        g.setColour (detecting ? pink : green);
        g.fillRoundedRectangle (keyTabAutoDetectRect, 6.0f);
        g.setColour (detecting ? bright : juce::Colours::black);
        g.setFont (juce::FontOptions (11.5f).withStyle ("Bold"));
        g.drawText (detecting ? "HUM ROOT NOTE NOW..." : ("AUTO ROOT DETECT (" + proc.autoRootStatusText + ")"),
                    keyTabAutoDetectRect, juce::Justification::centred);

        g.setColour (gold);
        g.setFont (juce::FontOptions (13.0f).withStyle ("Bold"));
        g.drawText ("SCALES & MAQAMAT (MANUAL SELECTION):", 365, 84, 520, 22, juce::Justification::centredLeft);

        for (int m = 0; m < 10; ++m)
        {
            maqamPresetRects[m] = { 365.0f + (m % 2) * 275.0f, 112.0f + (m / 2) * 44.0f, 260.0f, 38.0f };
            bool active = (proc.currentMaqamIndex == m);
            g.setColour (active ? gold.withAlpha (0.28f) : juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (maqamPresetRects[m], 6.0f);
            g.setColour (active ? gold : muted.withAlpha (0.35f));
            g.drawRoundedRectangle (maqamPresetRects[m], 6.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
            g.drawText (juce::String (kMaqamat[m].nameEn) + "  (" + juce::String (kMaqamat[m].descEn) + ")",
                        maqamPresetRects[m].reduced (10.0f), juce::Justification::centredLeft);
        }

        g.setColour (bright);
        g.setFont (juce::FontOptions (12.0f));
        g.drawText ("Quarter-Tone Depth: " + juce::String ((int) proc.quarterToneCents) + " cents", 365, 342, 230, 22, juce::Justification::centredLeft);

        qtDepthSliderRect = { 595.0f, 346.0f, 290.0f, 14.0f };
        g.setColour (juce::Colour (0xff16161a));
        g.fillRoundedRectangle (qtDepthSliderRect, 7.0f);
        float normQt = (proc.quarterToneCents - (-65.0f)) / 30.0f;
        g.setColour (gold);
        g.fillRoundedRectangle (qtDepthSliderRect.getX(), qtDepthSliderRect.getY(),
                                qtDepthSliderRect.getWidth() * juce::jlimit (0.0f, 1.0f, normQt), 14.0f, 7.0f);
    }

    void drawChordsTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::FontOptions (13.0f).withStyle ("Bold"));
        g.drawText ("SMART DEGREE CHORD MATRIX:", 48, 82, 340, 22, juce::Justification::centredLeft);

        g.setColour (green);
        g.setFont (juce::FontOptions (11.5f).withStyle ("Bold"));
        g.drawText ("Chord Lock (Anti-Roll): " + juce::String ((int) proc.chordLockMs) + " ms", 470, 82, 210, 22, juce::Justification::centredRight);
        chordLockSliderRect = { 690.0f, 87.0f, 230.0f, 12.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (chordLockSliderRect, 6.0f);
        float clNorm = juce::jlimit (0.0f, 1.0f, (proc.chordLockMs - 35.0f) / 165.0f);
        g.setColour (green);
        g.fillRoundedRectangle (chordLockSliderRect.getX(), chordLockSliderRect.getY(),
                                chordLockSliderRect.getWidth() * clNorm, 12.0f, 6.0f);

        const char* roman[7] = { "I", "II", "III", "IV", "V", "VI", "VII" };

        for (int deg = 0; deg < 7; ++deg)
        {
            int pc = proc.degreePitchClasses[deg];
            bool isQt = (proc.noteStates[pc] == 2);
            juce::Rectangle<float> col (48.0f + deg * 126.0f, 112.0f, 118.0f, 278.0f);

            g.setColour (juce::Colour (0xff1b1b20));
            g.fillRoundedRectangle (col, 7.0f);
            g.setColour (isQt ? gold : pink.withAlpha (0.65f));
            g.drawRoundedRectangle (col, 7.0f, 1.3f);

            g.setColour (bright);
            g.setFont (juce::FontOptions (13.0f).withStyle ("Bold"));
            g.drawText (juce::String (roman[deg]) + " - " + kNoteNames[pc] + (isQt ? " 1/2b" : ""),
                        col.removeFromTop (28.0f), juce::Justification::centred);

            const auto& opts = proc.degreeChordOptions[deg];
            int maxShow = juce::jmin (6, (int) opts.size());
            for (int v = 0; v < maxShow; ++v)
            {
                juce::Rectangle<float> btn (col.getX() + 6.0f, col.getY() + 4.0f + v * 40.0f, col.getWidth() - 12.0f, 34.0f);
                chordOptionRects[deg][v] = btn;
                bool isSel = (proc.selectedDegreeVoicing[deg] == v);

                g.setColour (isSel ? (isQt ? gold : green) : juce::Colour (0xff26262f));
                g.fillRoundedRectangle (btn, 5.0f);
                g.setColour (isSel ? juce::Colours::black : (v == 0 ? bright : muted));
                g.setFont (isSel ? juce::FontOptions (11.5f).withStyle ("Bold") : juce::FontOptions (11.5f));
                g.drawText (opts[v].label, btn, juce::Justification::centred);
            }
            for (int v = maxShow; v < 6; ++v) chordOptionRects[deg][v] = {};
        }
    }

    void drawAssignTab (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (gold);
        g.setFont (juce::FontOptions (13.0f).withStyle ("Bold"));
        g.drawText ("CONDENSER MIC ISOLATION & GLOBAL GATE:", 48, 84, 720, 22, juce::Justification::centredLeft);

        condenserToggleRect = { 48.0f, 118.0f, 290.0f, 38.0f };
        g.setColour (proc.condenserMode ? pink : juce::Colour (0xff2b2b34));
        g.fillRoundedRectangle (condenserToggleRect, 6.0f);
        g.setColour (bright);
        g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
        g.drawText (proc.condenserMode ? "MIC: CONDENSER (75Hz HP)" : "MIC: DYNAMIC (STANDARD)", condenserToggleRect, juce::Justification::centred);

        calibrateRoomBtnRect = { 355.0f, 118.0f, 290.0f, 38.0f };
        bool cal = proc.isCalibratingRoom.load();
        g.setColour (cal ? pink : green);
        g.fillRoundedRectangle (calibrateRoomBtnRect, 6.0f);
        g.setColour (juce::Colours::black);
        g.drawText (cal ? "MEASURING ROOM NOISE..." : "AUTO-CALIBRATE ROOM (1 SEC)", calibrateRoomBtnRect, juce::Justification::centred);

        g.setColour (gold);
        g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
        g.drawText ("Global Mic Noise Gate: " + juce::String ((int) (proc.globalGateLevel * 100.0f)) + "%",
                    48, 174, 540, 20, juce::Justification::centredLeft);

        assignGateSliderRect = { 48.0f, 198.0f, 597.0f, 16.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (assignGateSliderRect, 8.0f);
        g.setColour (pink);
        g.fillRoundedRectangle (assignGateSliderRect.getX(), assignGateSliderRect.getY(),
                                assignGateSliderRect.getWidth() * proc.globalGateLevel, 16.0f, 8.0f);

        g.setColour (bright);
        g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
        g.drawText ("MIDI Automation Out: Pitch Bend (-50c + Vibrato) | Mod (CC1) | Expression (CC11) | Glide (CC5/65)",
                    48, 236, 860, 22, juce::Justification::centredLeft);

        g.setColour (muted);
        g.setFont (juce::FontOptions (11.5f));
        g.drawText ("FL Studio Live Record: Record -> Notes and automation. Automation appears in Piano Roll Event Editor.",
                    48, 268, 880, 20, juce::Justification::centredLeft);
        g.drawText ("Studio One / Cubase / Reaper: Insert VoxNI on Mic Audio Track -> Set Instrument Track MIDI Input to VoxNI -> Record.",
                    48, 294, 880, 20, juce::Justification::centredLeft);
        g.setColour (gold);
        g.drawText ("VoxNI v" + juce::String (kPluginVersionString) + "  |  Manufactured by " + juce::String (kManufacturerName),
                    48, 330, 880, 20, juce::Justification::centredLeft);
    }

    void drawBottomSection (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        bool detecting = proc.isDetectingKey.load();
        autoRootBottomRect = { 114.0f, 414.0f, 94.0f, 25.0f };
        g.setColour (detecting ? pink : green);
        g.fillRoundedRectangle (autoRootBottomRect, 5.0f);
        g.setColour (detecting ? bright : juce::Colours::black);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText (detecting ? "LISTENING..." : "AUTO ROOT", autoRootBottomRect, juce::Justification::centred);

        rootCycleRect = { 214.0f, 414.0f, 62.0f, 25.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (rootCycleRect, 5.0f);
        g.setColour (gold);
        g.drawText ("Key: " + juce::String (kNoteNames[proc.rootNote]), rootCycleRect, juce::Justification::centred);

        maqamCycleRect = { 282.0f, 414.0f, 118.0f, 25.0f };
        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (maqamCycleRect, 5.0f);
        g.setColour (gold);
        g.drawText ("Scale: " + juce::String (kMaqamat[proc.currentMaqamIndex].nameEn), maqamCycleRect, juce::Justification::centred);

        rangePresetRect = { 406.0f, 414.0f, 174.0f, 25.0f };
        g.setColour (proc.vocalRangePreset > 0 ? pink : juce::Colour (0xff26262e));
        g.fillRoundedRectangle (rangePresetRect, 5.0f);
        g.setColour (bright);
        const char* rangeNames[6] = { "Range: Follow", "Range: Bass", "Range: Baritone", "Range: Tenor", "Range: Alto", "Range: Soprano" };
        g.drawText (rangeNames[proc.vocalRangePreset % 6], rangePresetRect, juce::Justification::centred);

        semiMinusRect = { 586.0f, 414.0f, 28.0f, 25.0f };
        semiDisplayRect = { 618.0f, 414.0f, 108.0f, 25.0f };
        semiPlusRect = { 730.0f, 414.0f, 28.0f, 25.0f };

        g.setColour (juce::Colour (0xff26262e));
        g.fillRoundedRectangle (semiMinusRect, 5.0f);
        g.fillRoundedRectangle (semiDisplayRect, 5.0f);
        g.fillRoundedRectangle (semiPlusRect, 5.0f);
        g.setColour (gold);
        g.drawText ("-", semiMinusRect, juce::Justification::centred);
        g.drawText ("Trans: " + juce::String (proc.semitoneTranspose > 0 ? "+" : "") + juce::String (proc.semitoneTranspose), semiDisplayRect, juce::Justification::centred);
        g.drawText ("+", semiPlusRect, juce::Justification::centred);

        g.setColour (green);
        g.setFont (juce::FontOptions (11.0f).withStyle ("Bold"));
        g.drawText ("HOST: " + juce::String (proc.hostBpm.load(), 1) + " BPM", 768, 414, 175, 25, juce::Justification::centredLeft);

        // Left Expression Wheels + Dedicated VIB & CC Switches + Mini Intensity Buttons
        pitchWheelRect = { 34.0f, 446.0f, 28.0f, 88.0f };
        levelWheelRect = { 70.0f, 446.0f, 28.0f, 88.0f };
        g.setColour (juce::Colour (0xff151519));
        g.fillRoundedRectangle (pitchWheelRect, 5.0f);
        g.fillRoundedRectangle (levelWheelRect, 5.0f);

        float pbNorm = (float) proc.currentPitchBendValue.load() / 16383.0f;
        g.setColour (proc.vibratoEnabled.load() ? pink : pink.withAlpha (0.45f));
        g.fillRoundedRectangle (pitchWheelRect.getX() + 3.0f, pitchWheelRect.getY() + (1.0f - pbNorm) * 72.0f, 22.0f, 14.0f, 3.0f);

        g.setColour (proc.sendAutomationCC.load() ? gold : gold.withAlpha (0.35f));
        g.fillRoundedRectangle (levelWheelRect.getX() + 3.0f, levelWheelRect.getY() + (1.0f - proc.vowelEnv) * 72.0f, 22.0f, 14.0f, 3.0f);

        // VIB Switch + Depth (S/M/D)
        vibSwitchRect = { 32.0f, 540.0f, 32.0f, 20.0f };
        vibDepthRect  = { 32.0f, 564.0f, 32.0f, 20.0f };
        bool vibOn = proc.vibratoEnabled.load();
        g.setColour (vibOn ? pink : juce::Colour (0xff282830));
        g.fillRoundedRectangle (vibSwitchRect, 4.0f);
        g.setColour (bright);
        g.setFont (juce::FontOptions (9.0f).withStyle ("Bold"));
        g.drawText ("VIB", vibSwitchRect, juce::Justification::centred);

        const char* vibLvl[3] = { "S", "M", "D" };
        g.setColour (juce::Colour (0xff222229));
        g.fillRoundedRectangle (vibDepthRect, 4.0f);
        g.setColour (vibOn ? gold : muted);
        g.drawText (vibLvl[proc.vibratoDepthMode.load() % 3], vibDepthRect, juce::Justification::centred);

        // CC Switch + Sensitivity (S/M/H)
        ccSwitchRect = { 68.0f, 540.0f, 32.0f, 20.0f };
        ccSensRect   = { 68.0f, 564.0f, 32.0f, 20.0f };
        bool ccOn = proc.sendAutomationCC.load();
        g.setColour (ccOn ? green : juce::Colour (0xff282830));
        g.fillRoundedRectangle (ccSwitchRect, 4.0f);
        g.setColour (ccOn ? juce::Colours::black : bright);
        g.drawText ("CC", ccSwitchRect, juce::Justification::centred);

        const char* ccLvl[3] = { "S", "M", "H" };
        g.setColour (juce::Colour (0xff222229));
        g.fillRoundedRectangle (ccSensRect, 4.0f);
        g.setColour (ccOn ? green : muted);
        g.drawText (ccLvl[proc.ccSensitivityMode.load() % 3], ccSensRect, juce::Justification::centred);

        // 36-Key Visual Keyboard
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

        // Bottom Controls Row
        synthCycleRect = { 114.0f, 560.0f, 210.0f, 28.0f };
        g.setColour (juce::Colour (0xff25252c));
        g.fillRoundedRectangle (synthCycleRect, 5.0f);
        g.setColour (gold);
        g.setFont (juce::FontOptions (11.0f).withStyle ("Bold"));
        const char* synthNames[4] = { "Synth: Electric Saz", "Synth: G-Clarinet", "Synth: Electric Oud", "Synth: Sub Bass" };
        g.drawText (synthNames[proc.synthPresetIndex % 4], synthCycleRect, juce::Justification::centred);

        intellMuteRect = { 338.0f, 560.0f, 165.0f, 28.0f };
        bool intMuted = proc.intellAudioMuted.load();
        g.setColour (intMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (intellMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (intMuted ? "Intell Audio: MUTED" : "Intell Audio: ON", intellMuteRect, juce::Justification::centred);

        trigMuteRect = { 515.0f, 560.0f, 165.0f, 28.0f };
        bool trMuted = proc.trigAudioMuted.load();
        g.setColour (trMuted ? pink : juce::Colour (0xff2b3a30));
        g.fillRoundedRectangle (trigMuteRect, 5.0f);
        g.setColour (bright);
        g.drawText (trMuted ? "Trig Audio: MUTED" : "Trig Audio: ON", trigMuteRect, juce::Justification::centred);

        brandFooterRect = { 692.0f, 560.0f, 252.0f, 28.0f };
        g.setColour (juce::Colour (0xff1f1f26));
        g.fillRoundedRectangle (brandFooterRect, 5.0f);
        g.setColour (gold.withAlpha (0.4f));
        g.drawRoundedRectangle (brandFooterRect, 5.0f, 1.0f);
        g.setColour (gold);
        g.setFont (juce::FontOptions (10.5f).withStyle ("Bold"));
        g.drawText ("VoxNI v" + juce::String (kPluginVersionString) + "  |  " + juce::String (kManufacturerName),
                    brandFooterRect, juce::Justification::centred);
    }

    void drawArabicTooltip (juce::Graphics& g, juce::Colour gold, juce::Colour bright)
    {
        juce::Font tipFont (juce::FontOptions (12.5f).withName ("Tahoma"));
        g.setFont (tipFont);

        const float boxW = 300.0f;
        const float boxH = 44.0f;
        float bx = juce::jlimit (12.0f, (float) getWidth() - boxW - 12.0f, tooltipMousePos.x - boxW * 0.5f);
        float by = tooltipMousePos.y - boxH - 14.0f;
        if (by < 44.0f) by = tooltipMousePos.y + 18.0f;

        juce::Rectangle<float> tipRect (bx, by, boxW, boxH);
        g.setColour (juce::Colour (0xf0131318));
        g.fillRoundedRectangle (tipRect, 7.0f);
        g.setColour (gold.withAlpha (0.85f));
        g.drawRoundedRectangle (tipRect, 7.0f, 1.2f);

        g.setColour (bright);
        g.drawFittedText (currentTooltipText, tipRect.reduced (10.0f, 4.0f).toNearestInt(), juce::Justification::centred, 2);
    }

    void drawAboutPopupModal (juce::Graphics& g, juce::Colour gold, juce::Colour pink, juce::Colour green, juce::Colour bright, juce::Colour muted)
    {
        g.setColour (juce::Colours::black.withAlpha (0.76f));
        g.fillRect (getLocalBounds());

        aboutCardRect = { 230.0f, 120.0f, 500.0f, 355.0f };
        g.setColour (juce::Colour (0xff1a1a20));
        g.fillRoundedRectangle (aboutCardRect, 12.0f);
        g.setColour (gold);
        g.drawRoundedRectangle (aboutCardRect, 12.0f, 1.8f);

        aboutCloseBtnRect = { aboutCardRect.getRight() - 36.0f, aboutCardRect.getY() + 10.0f, 26.0f, 26.0f };
        g.setColour (juce::Colour (0xff2b2b34));
        g.fillRoundedRectangle (aboutCloseBtnRect, 5.0f);
        g.setColour (bright);
        g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
        g.drawText ("X", aboutCloseBtnRect, juce::Justification::centred);

        if (logoImage.isValid())
            g.drawImageWithin (logoImage, (int) aboutCardRect.getX() + 140, (int) aboutCardRect.getY() + 26, 220, 74, juce::RectanglePlacement::centred);

        g.setColour (gold);
        g.setFont (juce::FontOptions (18.0f).withStyle ("Bold"));
        g.drawText ("VoxNI  v" + juce::String (kPluginVersionString),
                    (int) aboutCardRect.getX() + 20, (int) aboutCardRect.getY() + 108, 460, 26, juce::Justification::centred);

        g.setColour (bright);
        g.setFont (juce::FontOptions (12.5f));
        g.drawFittedText ("Real-Time Vocal-to-MIDI, Oriental Maqam Quarter-Tone, Smart Chords & Beatbox Engine.",
                          (int) aboutCardRect.getX() + 36, (int) aboutCardRect.getY() + 142, 428, 38, juce::Justification::centred, 2);

        g.setColour (gold.withAlpha (0.25f));
        g.drawHorizontalLine ((int) aboutCardRect.getY() + 194, aboutCardRect.getX() + 36.0f, aboutCardRect.getRight() - 36.0f);

        g.setColour (muted);
        g.setFont (juce::FontOptions (11.5f).withStyle ("Bold"));
        g.drawText ("MANUFACTURED BY", (int) aboutCardRect.getX() + 20, (int) aboutCardRect.getY() + 206, 460, 18, juce::Justification::centred);

        g.setColour (bright);
        g.setFont (juce::FontOptions (17.0f).withStyle ("Bold"));
        g.drawText (kManufacturerName, (int) aboutCardRect.getX() + 20, (int) aboutCardRect.getY() + 226, 460, 26, juce::Justification::centred);

        g.setColour (muted);
        g.setFont (juce::FontOptions (11.0f));
        g.drawText ("Copyright (c) 2026 " + juce::String (kManufacturerName) + ". All Rights Reserved.",
                    (int) aboutCardRect.getX() + 20, (int) aboutCardRect.getY() + 258, 460, 20, juce::Justification::centred);

        aboutLinkBtnRect = { aboutCardRect.getX() + 95.0f, aboutCardRect.getY() + 294.0f, 310.0f, 38.0f };
        bool hasLink = juce::String (kManufacturerUrl).isNotEmpty();
        g.setColour (hasLink ? green : juce::Colour (0xff2b2b34));
        g.fillRoundedRectangle (aboutLinkBtnRect, 7.0f);
        g.setColour (hasLink ? juce::Colours::black : gold);
        g.setFont (juce::FontOptions (12.0f).withStyle ("Bold"));
        g.drawText (hasLink ? "CONTACT & PURCHASE - MIDO MAZZIKA" : "OFFICIAL LINK COMING SOON - MIDO MAZZIKA",
                    aboutLinkBtnRect, juce::Justification::centred);
    }

    void updateArabicTooltip (juce::Point<float> pt)
    {
        tooltipMousePos = pt;
        juce::String tip;

        if (logoClickRect.contains (pt) || brandFooterRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"VoxNI v1.0.0 من تطوير Mido Mazzika — اضغط هنا لمعلومات الإصدار والمصنّع");
        else if (exclusiveModeRects[0].contains (pt))
            tip = juce::String::fromUTF8 (u8"مود اللحن فقط: يحوّل غناءك ودندنتك إلى نوتات MIDI بالربع تون والفيبراتو");
        else if (exclusiveModeRects[1].contains (pt))
            tip = juce::String::fromUTF8 (u8"مود الإيقاع فقط: يحوّل البيت بوكس لضربات درامز وتريجر على الـ 8 بادات");
        else if (exclusiveModeRects[2].contains (pt))
            tip = juce::String::fromUTF8 (u8"مود الكوردات فقط: يعزف كوردات شرقية وغربية ذكية تلقائياً مع صوتك");
        else if (presetNameRect.contains (pt) || presetPrevRect.contains (pt) || presetNextRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"قائمة البريسيتس: اختار من 4 إعدادات جاهزة أو بريسيتس محفوظة على جهازك");
        else if (presetSaveRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"حفظ الإعدادات الحالية كبريسيت جديد على جهازك لاستدعائه في أي مشروع");
        else if (midiRecButtonRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تسجيل الميدي داخلياً مع النوتات والأوتوميشن لسحبه مباشرة للـ DAW");
        else if (quantizeCycleRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تظبيط إيقاع النوتات المسجلة (Quantize) على 1/8 أو 1/16 أو إيقافه");
        else if (midiDragButtonRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"اسحب الزرار ده وارميه جوه الـ Piano Roll لنقل العزف المسجل فوراً");
        else if (globalGateMeterRect.expanded (6.0f, 0.0f).contains (pt))
            tip = juce::String::fromUTF8 (u8"بوابة عزل الضوضاء (Noise Gate): اسحب الخط الوردي لفوق لمنع شوشرة الغرفة");
        else if (vibSwitchRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تفعيل/إيقاف مرور الفيبراتو والعُرَب البشرية على عجلة الـ Pitch Bend");
        else if (vibDepthRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"شدة الفيبراتو: S ناعم | M متوسط | D عميق (ممتاز للناي والكمانجة)");
        else if (ccSwitchRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تفعيل/إيقاف إرسال أوتوميشن النفس وليفل الصوت (Mod CC1 و Expr CC11)");
        else if (ccSensRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"حساسية استجابة الإكسبريشن والنفس: S هادي | M متوسط | H عالي");
        else if (autoRootBottomRect.contains (pt) || (currentTab == 2 && keyTabAutoDetectRect.contains (pt)))
            tip = juce::String::fromUTF8 (u8"Auto Root: دندن نغمة الركوز وسيتعرف البلجن على مفتاح الأغنية تلقائياً");
        else if (rootCycleRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تغيير درجة الركوز (Key) للمقام أو السلم الحالي");
        else if (maqamCycleRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"التبديل اليدوي بين المقامات الشرقية والسلالم الغربية");
        else if (rangePresetRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"تحديد طبقة الأوكتاف (Bass / Tenor / Alto...) لمنع القفزات المفاجئة");
        else if (semiMinusRect.contains (pt) || semiPlusRect.contains (pt) || semiDisplayRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"رفع أو خفض النغمة بالـ Semitone (اضغط في النص للتصفير)");
        else if (intellMuteRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"كتم أو تشغيل الصوت الداخلي للميلودي والكوردات لسماع الآلة الخارجية فقط");
        else if (trigMuteRect.contains (pt))
            tip = juce::String::fromUTF8 (u8"كتم أو تشغيل صوت عينات البادات الداخلي مع استمرار إرسال الـ MIDI");
        else if (currentTab == 0)
        {
            if (pitchAnalyzerRect.contains (pt))
                tip = juce::String::fromUTF8 (u8"أنالايزر حي لحركة صوتك والفيبراتو والربع تون لحظياً بستايل Auto-Tune");
            else if (stickinessSliderRect.expanded (0.0f, 4.0f).contains (pt))
                tip = juce::String::fromUTF8 (u8"ثبات النوتة (Stickiness): يمنع اهتزاز الحرف وتبديله بالغلط مع الفيبراتو");
            else if (voiceModeRects[0].contains (pt))
                tip = juce::String::fromUTF8 (u8"MONO: يقطع الحرف القديم قبل الجديد لـ Piano Roll نضيف بدون تداخل");
            else if (voiceModeRects[1].contains (pt))
                tip = juce::String::fromUTF8 (u8"LEGATO: تداخل ناعم بين النوتات لتشغيل الجلايد في آلات السنث والناي");
            else if (voiceModeRects[2].contains (pt))
                tip = juce::String::fromUTF8 (u8"POLY: يترك النوتات ترن وتندمج مع بعضها مثل دواسة Sustain خفيفة");
            else if (glideSliderRect.expanded (0.0f, 4.0f).contains (pt))
                tip = juce::String::fromUTF8 (u8"سرعة الجلايد (Portamento CC5/65): يزحلق النغمة بنعومة للسنث الخارجي");
            else if (melodyLockSliderRect.expanded (0.0f, 4.0f).contains (pt))
                tip = juce::String::fromUTF8 (u8"Note Lock: يمنع التريم والرول السريع بين الحروف عند الانتقال من نوتة لأخرى");
            else if (velocityCurveBoxRect.contains (pt))
                tip = juce::String::fromUTF8 (u8"منحنى الدايناميك: اسحب لفوق أو لتحت لضبط قوة ضربات الميدي مع شدة صوتك");
            else if (fixedVelToggleRect.contains (pt))
                tip = juce::String::fromUTF8 (u8"تثبيت قوة كل النوتات على أقصى شدة (127) أو تركها دايناميك مع صوتك");
            else if (playCalibBtnRect.contains (pt))
                tip = juce::String::fromUTF8 (u8"معايرة ضوضاء الغرفة في ثانية واحدة لضبط بوابة المايك أوتوماتيكياً");
            else
            {
                for (int i = 0; i < 8; ++i)
                {
                    if (padLevelRects[i].contains (pt))
                    {
                        tip = juce::String::fromUTF8 (u8"مستوى وقوة الباد (LVL %): اسحب لضبط قوة الـ Velocity المرسلة للآلة التانية");
                        break;
                    }
                    if (padBounds[i].contains (pt))
                    {
                        tip = juce::String::fromUTF8 (u8"باد إيقاع: اضغط للتجربة أو اسحب ملف WAV/MP3 وأفلته هنا مباشرة");
                        break;
                    }
                }
            }
        }
        else if (currentTab == 1)
        {
            if (trainPadLevelSliderRect.expanded (0.0f, 4.0f).contains (pt))
                tip = juce::String::fromUTF8 (u8"التحكم في قوة الـ Velocity ومستوى صوت الباد المحدد بشكل مستقل");
            else if (retrigLockSliderRect.expanded (0.0f, 4.0f).contains (pt))
                tip = juce::String::fromUTF8 (u8"Retrig Lock: يمنع تكرار الضربة الواحدة مرتين بالغلط (Anti-Double Hit)");
        }
        else if (currentTab == 3 && chordLockSliderRect.expanded (0.0f, 4.0f).contains (pt))
        {
            tip = juce::String::fromUTF8 (u8"Chord Lock: يثبّت الكورد ويمنع تقلب الكوردات في النص أثناء الانتقال بصوتك");
        }

        if (tip != currentTooltipText)
        {
            currentTooltipText = tip;
            repaint();
        }
    }

    void mouseMove (const juce::MouseEvent& e) override
    {
        if (! showAboutPopup)
            updateArabicTooltip (e.position);
    }

    void mouseExit (const juce::MouseEvent&) override
    {
        if (currentTooltipText.isNotEmpty())
        {
            currentTooltipText.clear();
            repaint();
        }
    }

    void mouseDown (const juce::MouseEvent& e) override
    {
        juce::Point<float> pt = e.position;

        if (showAboutPopup)
        {
            if (aboutCloseBtnRect.contains (pt) || ! aboutCardRect.contains (pt))
            {
                showAboutPopup = false;
                repaint();
                return;
            }
            if (aboutLinkBtnRect.contains (pt))
            {
                juce::String urlStr (kManufacturerUrl);
                if (urlStr.isNotEmpty())
                    juce::URL (urlStr).launchInDefaultBrowser();
                return;
            }
            return;
        }

        if (logoClickRect.contains (pt) || brandFooterRect.contains (pt))
        {
            currentTooltipText.clear();
            showAboutPopup = true;
            repaint();
            return;
        }

        if (presetPrevRect.contains (pt))
        {
            int nextIdx = (proc.currentPresetIndex <= 0) ? ((int) kFactoryPresetNames.size() - 1) : (proc.currentPresetIndex - 1);
            proc.applyFactoryPreset (nextIdx);
            repaint(); return;
        }
        if (presetNextRect.contains (pt))
        {
            int nextIdx = (proc.currentPresetIndex + 1) % (int) kFactoryPresetNames.size();
            proc.applyFactoryPreset (nextIdx);
            repaint(); return;
        }
        if (presetNameRect.contains (pt)) { showPresetsMenu(); return; }
        if (presetSaveRect.contains (pt)) { openSavePresetDialog(); return; }

        if (globalGateMeterRect.expanded (6.0f, 0.0f).contains (pt))
        {
            proc.globalGateLevel = juce::jlimit (0.02f, 0.95f, (396.0f - pt.y) / 308.0f);
            repaint(); return;
        }

        for (int m = 0; m < 3; ++m)
            if (exclusiveModeRects[m].contains (pt)) { proc.setExclusiveEngineMode (m); repaint(); return; }

        if (midiRecButtonRect.contains (pt))
        {
            if (proc.isCapturingMidi.load()) proc.stopMidiCapture();
            else proc.startMidiCapture();
            repaint(); return;
        }
        if (quantizeCycleRect.contains (pt))   { proc.quantizeMode = (proc.quantizeMode + 1) % 3; repaint(); return; }
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

        if (autoRootBottomRect.contains (pt) || (currentTab == 2 && keyTabAutoDetectRect.contains (pt)))
        {
            proc.setExclusiveEngineMode (ModeMelodyOnly);
            proc.toggleAutoRootDetection();
            repaint(); return;
        }

        if (rootCycleRect.contains (pt))    { proc.applyMaqamPreset ((proc.rootNote + 1) % 12, proc.currentMaqamIndex); repaint(); return; }
        if (maqamCycleRect.contains (pt))   { proc.applyMaqamPreset (proc.rootNote, (proc.currentMaqamIndex + 1) % (int) kMaqamat.size()); repaint(); return; }
        if (rangePresetRect.contains (pt))  { proc.vocalRangePreset = (proc.vocalRangePreset + 1) % 6; repaint(); return; }
        if (semiMinusRect.contains (pt))    { proc.semitoneTranspose = juce::jlimit (-24, 24, proc.semitoneTranspose - 1); repaint(); return; }
        if (semiPlusRect.contains (pt))     { proc.semitoneTranspose = juce::jlimit (-24, 24, proc.semitoneTranspose + 1); repaint(); return; }
        if (semiDisplayRect.contains (pt))  { proc.semitoneTranspose = 0; repaint(); return; }

        if (vibSwitchRect.contains (pt))    { proc.vibratoEnabled.store (! proc.vibratoEnabled.load()); repaint(); return; }
        if (vibDepthRect.contains (pt))     { proc.vibratoDepthMode.store ((proc.vibratoDepthMode.load() + 1) % 3); repaint(); return; }
        if (ccSwitchRect.contains (pt))     { proc.sendAutomationCC.store (! proc.sendAutomationCC.load()); repaint(); return; }
        if (ccSensRect.contains (pt))       { proc.ccSensitivityMode.store ((proc.ccSensitivityMode.load() + 1) % 3); repaint(); return; }

        if (intellMuteRect.contains (pt))   { proc.intellAudioMuted.store (! proc.intellAudioMuted.load()); repaint(); return; }
        if (trigMuteRect.contains (pt))     { proc.trigAudioMuted.store (! proc.trigAudioMuted.load()); repaint(); return; }
        if (synthCycleRect.contains (pt))   { proc.synthPresetIndex = (proc.synthPresetIndex + 1) % 4; repaint(); return; }

        for (int k = 0; k < 36; ++k)
            if (pianoKeyRects[k].contains (pt)) { proc.setExclusiveEngineMode (ModeMelodyOnly); proc.guiManualMidiNote.store (48 + k); repaint(); return; }

        if (currentTab == 0)
        {
            if (playTrigSensRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.triggerSensitivity = juce::jlimit (0.05f, 1.0f, (pt.x - playTrigSensRect.getX()) / playTrigSensRect.getWidth());
                repaint(); return;
            }
            for (int v = 0; v < 3; ++v)
            {
                if (voiceModeRects[v].contains (pt))
                {
                    proc.setMelodyVoiceMode (v);
                    repaint(); return;
                }
            }
            if (glideSliderRect.expanded (0.0f, 5.0f).contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - glideSliderRect.getX()) / glideSliderRect.getWidth());
                proc.setGlideMs (norm * 300.0f);
                repaint(); return;
            }
            if (melodyLockSliderRect.expanded (0.0f, 5.0f).contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - melodyLockSliderRect.getX()) / melodyLockSliderRect.getWidth());
                proc.melodyNoteLockMs = 10.0f + norm * 80.0f;
                repaint(); return;
            }
            if (velocityCurveBoxRect.contains (pt))
            {
                float normY = 1.0f - juce::jlimit (0.0f, 1.0f, (pt.y - velocityCurveBoxRect.getY()) / velocityCurveBoxRect.getHeight());
                proc.velocityCurve = -1.0f + normY * 2.0f;
                proc.fixedVelocity127 = false;
                repaint(); return;
            }
            if (fixedVelToggleRect.contains (pt))
            {
                proc.fixedVelocity127 = ! proc.fixedVelocity127;
                repaint(); return;
            }
            if (playCalibBtnRect.contains (pt))
            {
                proc.startRoomCalibration();
                repaint(); return;
            }
            for (int i = 0; i < 8; ++i)
            {
                if (padLoadRects[i].contains (pt)) { openSampleChooserForPad (i); return; }
                if (padLevelRects[i].expanded (0.0f, 3.0f).contains (pt))
                {
                    draggingPadLevelIndex = i;
                    float norm = juce::jlimit (0.15f, 1.0f, (pt.x - padLevelRects[i].getX()) / padLevelRects[i].getWidth());
                    proc.pads[i].levelTrim = norm;
                    repaint(); return;
                }
                if (padBounds[i].contains (pt))    { proc.setExclusiveEngineMode (ModeTriggersOnly); proc.triggerPadNow (i); repaint(); return; }
            }
            if (stickinessSliderRect.expanded (0.0f, 5.0f).contains (pt))
            {
                proc.stickiness = juce::jlimit (0.0f, 1.0f, (pt.x - stickinessSliderRect.getX()) / stickinessSliderRect.getWidth());
                repaint(); return;
            }
            for (int i = 0; i < 12; ++i)
            {
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 22.0f)
                {
                    proc.setExclusiveEngineMode (ModeMelodyOnly);
                    if (e.mods.isRightButtonDown() || e.mods.isShiftDown()) { proc.noteStates[i] = (proc.noteStates[i] + 1) % 3; proc.rebuildSmartDegreeChords(); }
                    else proc.guiManualMidiNote.store (60 + i);
                    repaint(); return;
                }
            }
        }
        else if (currentTab == 1)
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
            if (trainSensSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.trainSensitivity = juce::jlimit (0.05f, 1.0f, (pt.x - trainSensSliderRect.getX()) / trainSensSliderRect.getWidth());
                repaint(); return;
            }
            if (hiHatBoostSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.hiHatBoost = juce::jlimit (0.0f, 1.0f, (pt.x - hiHatBoostSliderRect.getX()) / hiHatBoostSliderRect.getWidth());
                repaint(); return;
            }
            if (padSensSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.pads[selectedTrainPad].padSensitivity = juce::jlimit (0.1f, 1.0f, (pt.x - padSensSliderRect.getX()) / padSensSliderRect.getWidth());
                repaint(); return;
            }
            if (trainPadLevelSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.pads[selectedTrainPad].levelTrim = juce::jlimit (0.15f, 1.0f, (pt.x - trainPadLevelSliderRect.getX()) / trainPadLevelSliderRect.getWidth());
                repaint(); return;
            }
            if (retrigLockSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - retrigLockSliderRect.getX()) / retrigLockSliderRect.getWidth());
                proc.triggerRetrigLockMs = 40.0f + norm * 160.0f;
                repaint(); return;
            }
        }
        else if (currentTab == 2)
        {
            for (int m = 0; m < 10; ++m)
                if (maqamPresetRects[m].contains (pt)) { proc.applyMaqamPreset (proc.rootNote, m); repaint(); return; }

            if (qtDepthSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - qtDepthSliderRect.getX()) / qtDepthSliderRect.getWidth());
                proc.quarterToneCents = -65.0f + norm * 30.0f;
                repaint(); return;
            }
            for (int i = 0; i < 12; ++i)
                if (wheelSegmentCenters[i].getDistanceFrom (pt) < 24.0f)
                {
                    proc.noteStates[i] = (proc.noteStates[i] + 1) % 3;
                    proc.rebuildSmartDegreeChords();
                    repaint(); return;
                }
        }
        else if (currentTab == 3)
        {
            if (chordLockSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - chordLockSliderRect.getX()) / chordLockSliderRect.getWidth());
                proc.chordLockMs = 35.0f + norm * 165.0f;
                repaint(); return;
            }
            for (int deg = 0; deg < 7; ++deg)
            {
                for (int v = 0; v < 6; ++v)
                {
                    if (chordOptionRects[deg][v].contains (pt))
                    {
                        proc.selectedDegreeVoicing[deg] = v;
                        proc.auditionDegreeChord (deg);
                        repaint(); return;
                    }
                }
            }
        }
        else if (currentTab == 4)
        {
            if (condenserToggleRect.contains (pt))  { proc.condenserMode = ! proc.condenserMode; repaint(); return; }
            if (calibrateRoomBtnRect.contains (pt)) { proc.startRoomCalibration(); repaint(); return; }
            if (assignGateSliderRect.expanded (0.0f, 4.0f).contains (pt))
            {
                proc.globalGateLevel = juce::jlimit (0.02f, 0.95f, (pt.x - assignGateSliderRect.getX()) / assignGateSliderRect.getWidth());
                repaint(); return;
            }
        }
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (showAboutPopup) return;
        juce::Point<float> pt = e.position;

        if (draggingPadLevelIndex >= 0 && draggingPadLevelIndex < 8)
        {
            auto& r = padLevelRects[draggingPadLevelIndex];
            proc.pads[draggingPadLevelIndex].levelTrim = juce::jlimit (0.15f, 1.0f, (pt.x - r.getX()) / r.getWidth());
            repaint(); return;
        }

        if (globalGateMeterRect.expanded (6.0f, 0.0f).contains (e.mouseDownPosition))
        {
            proc.globalGateLevel = juce::jlimit (0.02f, 0.95f, (396.0f - pt.y) / 308.0f);
            repaint(); return;
        }

        if (currentTab == 0)
        {
            if (playTrigSensRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                proc.triggerSensitivity = juce::jlimit (0.05f, 1.0f, (pt.x - playTrigSensRect.getX()) / playTrigSensRect.getWidth());
                repaint(); return;
            }
            if (stickinessSliderRect.expanded (0.0f, 5.0f).contains (e.mouseDownPosition))
            {
                proc.stickiness = juce::jlimit (0.0f, 1.0f, (pt.x - stickinessSliderRect.getX()) / stickinessSliderRect.getWidth());
                repaint(); return;
            }
            if (glideSliderRect.expanded (0.0f, 5.0f).contains (e.mouseDownPosition))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - glideSliderRect.getX()) / glideSliderRect.getWidth());
                proc.setGlideMs (norm * 300.0f);
                repaint(); return;
            }
            if (melodyLockSliderRect.expanded (0.0f, 5.0f).contains (e.mouseDownPosition))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - melodyLockSliderRect.getX()) / melodyLockSliderRect.getWidth());
                proc.melodyNoteLockMs = 10.0f + norm * 80.0f;
                repaint(); return;
            }
            if (velocityCurveBoxRect.contains (e.mouseDownPosition))
            {
                float normY = 1.0f - juce::jlimit (0.0f, 1.0f, (pt.y - velocityCurveBoxRect.getY()) / velocityCurveBoxRect.getHeight());
                proc.velocityCurve = -1.0f + normY * 2.0f;
                proc.fixedVelocity127 = false;
                repaint(); return;
            }
        }
        else if (currentTab == 1)
        {
            if (trainSensSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                proc.trainSensitivity = juce::jlimit (0.05f, 1.0f, (pt.x - trainSensSliderRect.getX()) / trainSensSliderRect.getWidth());
                repaint(); return;
            }
            if (hiHatBoostSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                proc.hiHatBoost = juce::jlimit (0.0f, 1.0f, (pt.x - hiHatBoostSliderRect.getX()) / hiHatBoostSliderRect.getWidth());
                repaint(); return;
            }
            if (padSensSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                proc.pads[selectedTrainPad].padSensitivity = juce::jlimit (0.1f, 1.0f, (pt.x - padSensSliderRect.getX()) / padSensSliderRect.getWidth());
                repaint(); return;
            }
            if (trainPadLevelSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                proc.pads[selectedTrainPad].levelTrim = juce::jlimit (0.15f, 1.0f, (pt.x - trainPadLevelSliderRect.getX()) / trainPadLevelSliderRect.getWidth());
                repaint(); return;
            }
            if (retrigLockSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
            {
                float norm = juce::jlimit (0.0f, 1.0f, (pt.x - retrigLockSliderRect.getX()) / retrigLockSliderRect.getWidth());
                proc.triggerRetrigLockMs = 40.0f + norm * 160.0f;
                repaint(); return;
            }
        }
        else if (currentTab == 2 && qtDepthSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
        {
            float norm = juce::jlimit (0.0f, 1.0f, (pt.x - qtDepthSliderRect.getX()) / qtDepthSliderRect.getWidth());
            proc.quarterToneCents = -65.0f + norm * 30.0f;
            repaint(); return;
        }
        else if (currentTab == 3 && chordLockSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
        {
            float norm = juce::jlimit (0.0f, 1.0f, (pt.x - chordLockSliderRect.getX()) / chordLockSliderRect.getWidth());
            proc.chordLockMs = 35.0f + norm * 165.0f;
            repaint(); return;
        }
        else if (currentTab == 4 && assignGateSliderRect.expanded (0.0f, 4.0f).contains (e.mouseDownPosition))
        {
            proc.globalGateLevel = juce::jlimit (0.02f, 0.95f, (pt.x - assignGateSliderRect.getX()) / assignGateSliderRect.getWidth());
            repaint(); return;
        }

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
        draggingPadLevelIndex = -1;
        proc.guiManualMidiNote.store (-1);
    }

private:
    VoxN2AudioProcessor& proc;
    juce::Image logoImage;
    int currentTab = 0;
    int selectedTrainPad = 0;
    int draggingPadLevelIndex = -1;
    bool isDraggingMidi = false;
    bool showAboutPopup = false;

    juce::String currentTooltipText;
    juce::Point<float> tooltipMousePos { 480.0f, 300.0f };

    std::array<juce::Rectangle<float>, 3> exclusiveModeRects;
    std::array<juce::Rectangle<float>, 3> voiceModeRects;
    std::array<juce::Rectangle<float>, 5> tabRects;
    std::array<juce::Rectangle<float>, 8> padBounds;
    std::array<juce::Rectangle<float>, 8> padLoadRects;
    std::array<juce::Rectangle<float>, 8> padLevelRects;
    std::array<juce::Point<float>, 12> wheelSegmentCenters;
    std::array<juce::Rectangle<float>, 10> maqamPresetRects;
    std::array<std::array<juce::Rectangle<float>, 6>, 7> chordOptionRects;
    std::array<juce::Rectangle<float>, 36> pianoKeyRects;

    juce::Rectangle<float> logoClickRect, brandFooterRect;
    juce::Rectangle<float> presetPrevRect, presetNameRect, presetNextRect, presetSaveRect;
    juce::Rectangle<float> aboutCardRect, aboutCloseBtnRect, aboutLinkBtnRect;
    juce::Rectangle<float> globalGateMeterRect, pitchAnalyzerRect;
    juce::Rectangle<float> midiRecButtonRect, quantizeCycleRect, midiDragButtonRect, midiClearButtonRect;
    juce::Rectangle<float> playTrigSensRect, stickinessSliderRect, glideSliderRect, melodyLockSliderRect, velocityCurveBoxRect, fixedVelToggleRect, playCalibBtnRect;
    juce::Rectangle<float> keyTabAutoDetectRect, qtDepthSliderRect;
    juce::Rectangle<float> trainVoiceBtnRect, trainLoadButtonRect, trainPlayButtonRect, trainRevertButtonRect;
    juce::Rectangle<float> trainSensSliderRect, hiHatBoostSliderRect, padSensSliderRect, trainPadLevelSliderRect, retrigLockSliderRect;
    juce::Rectangle<float> chordLockSliderRect;
    juce::Rectangle<float> condenserToggleRect, calibrateRoomBtnRect, assignGateSliderRect;
    juce::Rectangle<float> autoRootBottomRect, rootCycleRect, maqamCycleRect, rangePresetRect, semiMinusRect, semiDisplayRect, semiPlusRect;
    juce::Rectangle<float> pitchWheelRect, levelWheelRect, vibSwitchRect, vibDepthRect, ccSwitchRect, ccSensRect;
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
