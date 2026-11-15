cmake_minimum_required(VERSION 3.22)
project(VOXN2_CyberGold VERSION 2.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        7.0.12
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(VOXN2_CyberGold
    COMPANY_NAME "VoxN Studio"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Voxn
    PLUGIN_CODE VxN2
    FORMATS VST3 Standalone
    PRODUCT_NAME "VOXN 2 Cyber Gold"
)

target_sources(VOXN2_CyberGold
    PRIVATE
        Source/VoxN2Plugin.cpp
)

target_compile_definitions(VOXN2_CyberGold
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_USE_MP3AUDIOFORMAT=1
        JUCE_MODAL_LOOPS_PERMITTED=1
)

target_link_libraries(VOXN2_CyberGold
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
)
