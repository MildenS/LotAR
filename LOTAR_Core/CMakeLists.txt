cmake_minimum_required(VERSION 3.27.4)

set(CORE_PROJECT_NAME LOTAR_Core)


find_package(OpenCV REQUIRED)


add_executable(${CORE_PROJECT_NAME}
	src/main.cpp
)


target_link_libraries(${CORE_PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_include_directories(${CORE_PROJECT_NAME} PRIVATE includes)
target_include_directories(${CORE_PROJECT_NAME} PRIVATE src)