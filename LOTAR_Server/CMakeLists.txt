cmake_minimum_required(VERSION 3.27.4)

set(SERVER_PROJECT_NAME LOTAR_Server)


find_package(asio REQUIRED)


add_executable(${SERVER_PROJECT_NAME}
	src/main.cpp
)


target_link_libraries(${SERVER_PROJECT_NAME} PRIVATE asio::asio)
target_include_directories(${SERVER_PROJECT_NAME} PRIVATE includes)
target_include_directories(${SERVER_PROJECT_NAME} PRIVATE src)