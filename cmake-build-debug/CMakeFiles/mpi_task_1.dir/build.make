# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2019.3.4\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2019.3.4\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Blind\CLionProjects\mpi_task_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mpi_task_1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_task_1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_task_1.dir/flags.make

CMakeFiles/mpi_task_1.dir/main.obj: CMakeFiles/mpi_task_1.dir/flags.make
CMakeFiles/mpi_task_1.dir/main.obj: CMakeFiles/mpi_task_1.dir/includes_C.rsp
CMakeFiles/mpi_task_1.dir/main.obj: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_task_1.dir/main.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\mpi_task_1.dir\main.obj   -c C:\Users\Blind\CLionProjects\mpi_task_2\main.c

CMakeFiles/mpi_task_1.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_task_1.dir/main.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Blind\CLionProjects\mpi_task_2\main.c > CMakeFiles\mpi_task_1.dir\main.i

CMakeFiles/mpi_task_1.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_task_1.dir/main.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Blind\CLionProjects\mpi_task_2\main.c -o CMakeFiles\mpi_task_1.dir\main.s

CMakeFiles/mpi_task_1.dir/methods.obj: CMakeFiles/mpi_task_1.dir/flags.make
CMakeFiles/mpi_task_1.dir/methods.obj: CMakeFiles/mpi_task_1.dir/includes_C.rsp
CMakeFiles/mpi_task_1.dir/methods.obj: ../methods.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/mpi_task_1.dir/methods.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\mpi_task_1.dir\methods.obj   -c C:\Users\Blind\CLionProjects\mpi_task_2\methods.c

CMakeFiles/mpi_task_1.dir/methods.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_task_1.dir/methods.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\Blind\CLionProjects\mpi_task_2\methods.c > CMakeFiles\mpi_task_1.dir\methods.i

CMakeFiles/mpi_task_1.dir/methods.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_task_1.dir/methods.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\Blind\CLionProjects\mpi_task_2\methods.c -o CMakeFiles\mpi_task_1.dir\methods.s

# Object files for target mpi_task_1
mpi_task_1_OBJECTS = \
"CMakeFiles/mpi_task_1.dir/main.obj" \
"CMakeFiles/mpi_task_1.dir/methods.obj"

# External object files for target mpi_task_1
mpi_task_1_EXTERNAL_OBJECTS =

mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/main.obj
mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/methods.obj
mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/build.make
mpi_task_1.exe: C:/Program\ Files\ (x86)/Microsoft\ SDKs/MPI/Lib/x64/msmpi.lib
mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/linklibs.rsp
mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/objects1.rsp
mpi_task_1.exe: CMakeFiles/mpi_task_1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable mpi_task_1.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\mpi_task_1.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_task_1.dir/build: mpi_task_1.exe

.PHONY : CMakeFiles/mpi_task_1.dir/build

CMakeFiles/mpi_task_1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\mpi_task_1.dir\cmake_clean.cmake
.PHONY : CMakeFiles/mpi_task_1.dir/clean

CMakeFiles/mpi_task_1.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\Blind\CLionProjects\mpi_task_2 C:\Users\Blind\CLionProjects\mpi_task_2 C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug C:\Users\Blind\CLionProjects\mpi_task_2\cmake-build-debug\CMakeFiles\mpi_task_1.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_task_1.dir/depend
