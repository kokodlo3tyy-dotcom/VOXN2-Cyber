cmake_minimum_required(VERSION 3.22)
project(VOXN2CyberGold VERSION 2.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(JUCE)

juce_add_plugin(VOXN2CyberGold
    COMPANY_NAME "VOXN Audio"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Voxn
    PLUGIN_CODE Vxn2
    FORMATS VST3
    PRODUCT_NAME "VOXN 2 Cyber Gold"
)

target_sources(VOXN2CyberGold
    PRIVATE
        Source/VoxN2Plugin.cpp
)

target_compile_definitions(VOXN2CyberGold
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(VOXN2CyberGold
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
)
