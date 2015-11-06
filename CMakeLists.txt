cmake_minimum_required(VERSION 2.8.3)

project(undistorter)
find_package(catkin_simple REQUIRED)

find_package(cmake_modules REQUIRED)
find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED) 

include_directories(include ${Eigen_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}_example src/example.cc)
target_link_libraries(${PROJECT_NAME}_example ${OpenCV_LIBS} ${catkin_LIBRARIES})

cs_install()
cs_export()
