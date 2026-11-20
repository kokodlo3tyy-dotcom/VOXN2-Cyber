cmake_minimum_required(VERSION 3.22)
project(VoxNI VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(JUCE)

juce_add_plugin(VoxNI
    COMPANY_NAME "Mido Mazzika"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Mdmz
    PLUGIN_CODE Vx01
    FORMATS VST3
    PRODUCT_NAME "VoxNI"
)

juce_add_binary_data(VoxNIBinaryData
    SOURCES
        voxni_logo.png
)

target_sources(VoxNI
    PRIVATE
        Source/VoxN2Plugin.cpp
)

target_compile_options(VoxNI
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

target_compile_definitions(VoxNI
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(VoxNI
    PRIVATE
        VoxNIBinaryData
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
)
