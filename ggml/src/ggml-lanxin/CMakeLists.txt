message(STATUS "Including the Lanxin PACC backend")

ggml_add_backend_library(ggml-lanxin
    ggml-lanxin.cpp
    ../../include/ggml-lanxin.h
)

target_compile_definitions(ggml-lanxin PRIVATE GGML_LANXIN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml-lanxin PRIVATE dl)
endif()
