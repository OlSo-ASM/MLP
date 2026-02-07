cmake_minimum_required(VERSION 3.15)
project(MLP VERSION 1.0 LANGUAGES CXX)

# c++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# sources
file(GLOB SOURCES "src/*.cpp")

#exe
add_executable(${PROJECT_NAME} ${SOURCES})

# include
target_include_directories(${PROJECT_NAME} PRIVATE
    include
    external/eigen-5.0.0   # Eigen заголовки
    external/SFML-3.0.2/include  # SFML заголовки
)

# sfml static linking
add_compile_definitions(SFML_STATIC)

target_link_libraries(MLP PRIVATE
    ../external/SFML-3.0.2/lib/sfml-graphics-s
    ../external/SFML-3.0.2/lib/sfml-window-s
    ../external/SFML-3.0.2/lib/sfml-system-s
    ../external/SFML-3.0.2/lib/freetype
)

# Windows system libraries for SFML
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        opengl32 winmm gdi32
    )
endif()
