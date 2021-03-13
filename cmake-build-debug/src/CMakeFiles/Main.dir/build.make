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
include src/CMakeFiles/Main.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/Main.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/Main.dir/flags.make

src/CMakeFiles/Main.dir/Network.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/Network.cpp.o: ../src/Network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/Main.dir/Network.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/Network.cpp.o -c "/home/kate/Neural Nets/Network/src/Network.cpp"

src/CMakeFiles/Main.dir/Network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/Network.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/Network.cpp" > CMakeFiles/Main.dir/Network.cpp.i

src/CMakeFiles/Main.dir/Network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/Network.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/Network.cpp" -o CMakeFiles/Main.dir/Network.cpp.s

src/CMakeFiles/Main.dir/HiddenLayer.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/HiddenLayer.cpp.o: ../src/HiddenLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/Main.dir/HiddenLayer.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/HiddenLayer.cpp.o -c "/home/kate/Neural Nets/Network/src/HiddenLayer.cpp"

src/CMakeFiles/Main.dir/HiddenLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/HiddenLayer.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/HiddenLayer.cpp" > CMakeFiles/Main.dir/HiddenLayer.cpp.i

src/CMakeFiles/Main.dir/HiddenLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/HiddenLayer.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/HiddenLayer.cpp" -o CMakeFiles/Main.dir/HiddenLayer.cpp.s

src/CMakeFiles/Main.dir/InputLayer.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/InputLayer.cpp.o: ../src/InputLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/Main.dir/InputLayer.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/InputLayer.cpp.o -c "/home/kate/Neural Nets/Network/src/InputLayer.cpp"

src/CMakeFiles/Main.dir/InputLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/InputLayer.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/InputLayer.cpp" > CMakeFiles/Main.dir/InputLayer.cpp.i

src/CMakeFiles/Main.dir/InputLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/InputLayer.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/InputLayer.cpp" -o CMakeFiles/Main.dir/InputLayer.cpp.s

src/CMakeFiles/Main.dir/Layer.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/Layer.cpp.o: ../src/Layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/Main.dir/Layer.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/Layer.cpp.o -c "/home/kate/Neural Nets/Network/src/Layer.cpp"

src/CMakeFiles/Main.dir/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/Layer.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/Layer.cpp" > CMakeFiles/Main.dir/Layer.cpp.i

src/CMakeFiles/Main.dir/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/Layer.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/Layer.cpp" -o CMakeFiles/Main.dir/Layer.cpp.s

src/CMakeFiles/Main.dir/NeuralNetwork.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/NeuralNetwork.cpp.o: ../src/NeuralNetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/Main.dir/NeuralNetwork.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/NeuralNetwork.cpp.o -c "/home/kate/Neural Nets/Network/src/NeuralNetwork.cpp"

src/CMakeFiles/Main.dir/NeuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/NeuralNetwork.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/NeuralNetwork.cpp" > CMakeFiles/Main.dir/NeuralNetwork.cpp.i

src/CMakeFiles/Main.dir/NeuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/NeuralNetwork.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/NeuralNetwork.cpp" -o CMakeFiles/Main.dir/NeuralNetwork.cpp.s

src/CMakeFiles/Main.dir/OutputLayer.cpp.o: src/CMakeFiles/Main.dir/flags.make
src/CMakeFiles/Main.dir/OutputLayer.cpp.o: ../src/OutputLayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/Main.dir/OutputLayer.cpp.o"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/OutputLayer.cpp.o -c "/home/kate/Neural Nets/Network/src/OutputLayer.cpp"

src/CMakeFiles/Main.dir/OutputLayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/OutputLayer.cpp.i"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/kate/Neural Nets/Network/src/OutputLayer.cpp" > CMakeFiles/Main.dir/OutputLayer.cpp.i

src/CMakeFiles/Main.dir/OutputLayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/OutputLayer.cpp.s"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/kate/Neural Nets/Network/src/OutputLayer.cpp" -o CMakeFiles/Main.dir/OutputLayer.cpp.s

# Object files for target Main
Main_OBJECTS = \
"CMakeFiles/Main.dir/Network.cpp.o" \
"CMakeFiles/Main.dir/HiddenLayer.cpp.o" \
"CMakeFiles/Main.dir/InputLayer.cpp.o" \
"CMakeFiles/Main.dir/Layer.cpp.o" \
"CMakeFiles/Main.dir/NeuralNetwork.cpp.o" \
"CMakeFiles/Main.dir/OutputLayer.cpp.o"

# External object files for target Main
Main_EXTERNAL_OBJECTS =

src/Main: src/CMakeFiles/Main.dir/Network.cpp.o
src/Main: src/CMakeFiles/Main.dir/HiddenLayer.cpp.o
src/Main: src/CMakeFiles/Main.dir/InputLayer.cpp.o
src/Main: src/CMakeFiles/Main.dir/Layer.cpp.o
src/Main: src/CMakeFiles/Main.dir/NeuralNetwork.cpp.o
src/Main: src/CMakeFiles/Main.dir/OutputLayer.cpp.o
src/Main: src/CMakeFiles/Main.dir/build.make
src/Main: src/CMakeFiles/Main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/kate/Neural Nets/Network/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable Main"
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/Main.dir/build: src/Main

.PHONY : src/CMakeFiles/Main.dir/build

src/CMakeFiles/Main.dir/clean:
	cd "/home/kate/Neural Nets/Network/cmake-build-debug/src" && $(CMAKE_COMMAND) -P CMakeFiles/Main.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/Main.dir/clean

src/CMakeFiles/Main.dir/depend:
	cd "/home/kate/Neural Nets/Network/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/kate/Neural Nets/Network" "/home/kate/Neural Nets/Network/src" "/home/kate/Neural Nets/Network/cmake-build-debug" "/home/kate/Neural Nets/Network/cmake-build-debug/src" "/home/kate/Neural Nets/Network/cmake-build-debug/src/CMakeFiles/Main.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/CMakeFiles/Main.dir/depend
