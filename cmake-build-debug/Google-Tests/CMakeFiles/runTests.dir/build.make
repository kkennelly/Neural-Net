# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = "/home/kate/Neural Nets/clion-2020.3.2/bin/cmake/linux/bin/cmake"

# The command to remove a file.
RM = "/home/kate/Neural Nets/clion-2020.3.2/bin/cmake/linux/bin/cmake" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/kate/Neural Nets/Network"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/kate/Neural Nets/Network/cmake-build-debug"

# Include any dependencies generated for this target.
include Google-Tests/CMakeFiles/runTests.dir/depend.make

# Include the progress variables for this target.
include Google-Tests/CMakeFiles/runTests.dir/progress.make

# Include the compile flags for this target's objects.
include Google-Tests/CMakeFiles/runTests.dir/flags.make

Google-Tests/CMakeFiles/runTests.dir/test.cpp.o: Google-Tests/CMakeFiles/runTests.dir/flags.make
Google-Tests/CMakeFiles/runTests.dir/test.cpp.o: ../Google-Tests/test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Google-Tests/CMakeFiles/runTests.dir/test.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/runTests.dir/test.cpp.o -c "/home/kate/Neural Nets/Network/Google-Tests/test.cpp"

Google-Tests/CMakeFiles/runTests.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runTests.dir/test.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/Google-Tests/test.cpp" > CMakeFiles/runTests.dir/test.cpp.i

Google-Tests/CMakeFiles/runTests.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runTests.dir/test.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/Google-Tests/test.cpp" -o CMakeFiles/runTests.dir/test.cpp.s

# Object files for target runTests
runTests_OBJECTS = \
"CMakeFiles/runTests.dir/test.cpp.o"

# External object files for target runTests
runTests_EXTERNAL_OBJECTS =

Google-Tests/runTests: Google-Tests/CMakeFiles/runTests.dir/test.cpp.o
Google-Tests/runTests: Google-Tests/CMakeFiles/runTests.dir/build.make
Google-Tests/runTests: /usr/lib/x86_64-linux-gnu/libgtest.a
Google-Tests/runTests: Google-Tests/CMakeFiles/runTests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable runTests"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runTests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Google-Tests/CMakeFiles/runTests.dir/build: Google-Tests/runTests

.PHONY : Google-Tests/CMakeFiles/runTests.dir/build

Google-Tests/CMakeFiles/runTests.dir/clean:
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" && $(CMAKE_COMMAND) -P CMakeFiles/runTests.dir/cmake_clean.cmake
.PHONY : Google-Tests/CMakeFiles/runTests.dir/clean

Google-Tests/CMakeFiles/runTests.dir/depend:
	cd "/home/kate/Neural Nets/Network/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/kate/Neural Nets/Network" "/home/kate/Neural Nets/Network/Google-Tests" "/home/kate/Neural Nets/Network/cmake-build-debug" "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests" "/home/kate/Neural Nets/Network/cmake-build-debug/Google-Tests/CMakeFiles/runTests.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : Google-Tests/CMakeFiles/runTests.dir/depend

