# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chaycv/wkcatkin/trackingobject/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chaycv/wkcatkin/trackingobject/build

# Include any dependencies generated for this target.
include tracking_object/CMakeFiles/tracking.dir/depend.make

# Include the progress variables for this target.
include tracking_object/CMakeFiles/tracking.dir/progress.make

# Include the compile flags for this target's objects.
include tracking_object/CMakeFiles/tracking.dir/flags.make

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o: tracking_object/CMakeFiles/tracking.dir/flags.make
tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o: /home/chaycv/wkcatkin/trackingobject/src/tracking_object/src/tracking.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/chaycv/wkcatkin/trackingobject/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o"
	cd /home/chaycv/wkcatkin/trackingobject/build/tracking_object && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/tracking.cpp.o -c /home/chaycv/wkcatkin/trackingobject/src/tracking_object/src/tracking.cpp

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/tracking.cpp.i"
	cd /home/chaycv/wkcatkin/trackingobject/build/tracking_object && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/chaycv/wkcatkin/trackingobject/src/tracking_object/src/tracking.cpp > CMakeFiles/tracking.dir/src/tracking.cpp.i

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/tracking.cpp.s"
	cd /home/chaycv/wkcatkin/trackingobject/build/tracking_object && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/chaycv/wkcatkin/trackingobject/src/tracking_object/src/tracking.cpp -o CMakeFiles/tracking.dir/src/tracking.cpp.s

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.requires:
.PHONY : tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.requires

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.provides: tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.requires
	$(MAKE) -f tracking_object/CMakeFiles/tracking.dir/build.make tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.provides.build
.PHONY : tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.provides

tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.provides.build: tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o

# Object files for target tracking
tracking_OBJECTS = \
"CMakeFiles/tracking.dir/src/tracking.cpp.o"

# External object files for target tracking
tracking_EXTERNAL_OBJECTS =

/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: tracking_object/CMakeFiles/tracking.dir/build.make
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libcv_bridge.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libimage_transport.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libmessage_filters.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libtinyxml.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libclass_loader.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/libPocoFoundation.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libdl.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libroslib.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_videostab.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_video.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_superres.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_stitching.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_photo.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_ocl.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_objdetect.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_nonfree.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_ml.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_legacy.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_imgproc.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_highgui.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_gpu.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_flann.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_features2d.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_core.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_contrib.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_calib3d.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libroscpp.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/librosconsole.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/liblog4cxx.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libxmlrpcpp.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libroscpp_serialization.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/librostime.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /opt/ros/indigo/lib/libcpp_common.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_nonfree.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_ocl.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_gpu.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_photo.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_objdetect.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_legacy.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_video.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_ml.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_calib3d.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_features2d.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_highgui.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_imgproc.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_flann.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: /home/chaycv/OpenCV/build/lib/libopencv_core.so.2.4.13
/home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking: tracking_object/CMakeFiles/tracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable /home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking"
	cd /home/chaycv/wkcatkin/trackingobject/build/tracking_object && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tracking_object/CMakeFiles/tracking.dir/build: /home/chaycv/wkcatkin/trackingobject/devel/lib/tracking_object/tracking
.PHONY : tracking_object/CMakeFiles/tracking.dir/build

tracking_object/CMakeFiles/tracking.dir/requires: tracking_object/CMakeFiles/tracking.dir/src/tracking.cpp.o.requires
.PHONY : tracking_object/CMakeFiles/tracking.dir/requires

tracking_object/CMakeFiles/tracking.dir/clean:
	cd /home/chaycv/wkcatkin/trackingobject/build/tracking_object && $(CMAKE_COMMAND) -P CMakeFiles/tracking.dir/cmake_clean.cmake
.PHONY : tracking_object/CMakeFiles/tracking.dir/clean

tracking_object/CMakeFiles/tracking.dir/depend:
	cd /home/chaycv/wkcatkin/trackingobject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chaycv/wkcatkin/trackingobject/src /home/chaycv/wkcatkin/trackingobject/src/tracking_object /home/chaycv/wkcatkin/trackingobject/build /home/chaycv/wkcatkin/trackingobject/build/tracking_object /home/chaycv/wkcatkin/trackingobject/build/tracking_object/CMakeFiles/tracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tracking_object/CMakeFiles/tracking.dir/depend

