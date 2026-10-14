cmake_minimum_required(VERSION 3.16)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS minimum version")
project(rbshift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(rbshift main.cpp vendor/rubberband/single/RubberBandSingle.cpp)
target_include_directories(rbshift PRIVATE vendor/rubberband)

if (MSVC)
  target_compile_options(rbshift PRIVATE /O2 /EHsc /utf-8 /wd4244 /wd4267 /wd4305)
  target_compile_definitions(rbshift PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
  # 정적 런타임: VC++ 재배포 패키지 없이 실행
  set_property(TARGET rbshift PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  target_compile_options(rbshift PRIVATE -O3)
endif()

if (APPLE)
  target_link_libraries(rbshift PRIVATE "-framework Accelerate")
elseif (UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(rbshift PRIVATE Threads::Threads)
endif()
