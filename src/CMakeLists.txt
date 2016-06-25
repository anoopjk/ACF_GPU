cmake_minimum_required(VERSION 2.8)
project(acf_detection)
find_package( OpenCV REQUIRED )
find_package( CUDA QUIET REQUIRED )
#list(APPEND CUDA_NVCC_FLAGS "-g -G")
#include_directories( /home/anoop/Documents/anoop_acf_detector/ )
#set(SOURCES main.cpp acf_detect.cpp  ${YOUR_DIRECTORY}/file1.h ${YOUR_DIRECTORY}/file2.h)


#add_executable(test ${SOURCES})

cuda_add_executable( acf_detection main.cpp  acf_detect.cpp  img_process.cu  acf_detect.hpp img_process.hpp cartToPolar.hpp)

target_link_libraries(acf_detection ${OpenCV_LIBS} )

