cmake_minimum_required (VERSION 2.6)
project (linergb_hsv_8bit)
SET(MAJOR_VERSION 0)
SET(MINOR_VERSION 2)
SET(PATCH_VERSION 0)
include_directories("."
	"${CMAKE_SOURCE_DIR}/include"
	)


#add_subdirectory(config)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeScripts)
set(CMAKE_INCLUDE_DIRS_CONFIGCMAKE ${CMAKE_SOURCE_DIR}/include)


set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "-ltcmalloc")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
set(CMAKE_C_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_MINSIZEREL "" CACHE INTERNAL "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "" CACHE INTERNAL "" FORCE)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)



if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE CMAKE_GCC_VERSION_FULL
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Typical output in CMAKE__GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
    #  Look for the version number
    string(REGEX MATCH "[0-9].[0-9].[0-9]" CMAKE_GCC_REGEX_VERSION "${CMAKE_GCC_VERSION_FULL}")

    # Split the three parts:
    string(REGEX MATCHALL "[0-9]" CMAKE_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

    list(GET CMAKE_GCC_VERSIONS 0 CMAKE_GCC_VERSION_MAJOR)
    list(GET CMAKE_GCC_VERSIONS 1 CMAKE_GCC_VERSION_MINOR)
    
    set(CMAKE_GCC_VERSION ${CMAKE_GCC_VERSION_MAJOR}${CMAKE_GCC_VERSION_MINOR})
    math(EXPR CMAKE_GCC_VERSION_NUM "${CMAKE_GCC_VERSION_MAJOR}*100 + ${CMAKE_GCC_VERSION_MINOR}")
    message(STATUS "Detected version of GNU GCC: ${CMAKE_GCC_VERSION} (${CMAKE_GCC_VERSION_NUM})")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling compiler (Add flags: -lprofiler)")
    set(USE_OMIT_FRAME_POINTER ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*)
        # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
    endif()
    
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
        set(X86_64 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(X86_64 1)
    endif()
    
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i386*)
        set(X86 1)
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86*)
        set(X86 1)
    endif()
        
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(ENABLE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()
    
    if(X86 OR X86_64)
        # enable everything, since the available set of instructions is checked at runtime
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH ON CACHE BOOL "Enable -ffast-math for GCC")
        set(ENABLE_SSE ON CACHE BOOL "Enable SSE for GCC")
        set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        set(ENABLE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
        set(ENABLE_SSSE3 ON CACHE BOOL "Enable SSSE3 for GCC")
        #set(ENABLE_SSE4_1 OFF CACHE BOOL "Enable SSE4.1 for GCC")
    endif()
endif()

set_target_properties( ${TargetName} PROPERTIES COMPILE_FLAGS -m32 LINK_FLAGS -m32 )

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

if(X86)
    if(NOT MINGW64)
	if(NOT X86_64)
	    if(NOT APPLE)
	#	set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -march=i686")
	    endif()
	endif()
    endif()
endif()

set(GTK_INTERFACE OFF CACHE BOOL "Enable GTK INTERFACE for Impulso")


set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -march=native -mtune=native -ggdb3 ")
if(USE_OMIT_FRAME_POINTER)   
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
endif()
if(USE_O2)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
endif()
if(USE_O3)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
endif()
if(USE_FAST_MATH)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
endif()
if(ENABLE_POWERPC)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
endif()
if(ENABLE_SSE)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse")
endif()
if(ENABLE_SSE2)
   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse2")
endif()
# SSE3 and further should be disabled under MingW because it generates compiler errors
if(NOT MINGW)
   if(ENABLE_SSE3)
      set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse3")
   endif()
   if(${CMAKE_GCC_VERSION_NUM} GREATER 402)
      set(HAVE_GCC43_OR_NEWER 1)
   endif()
   if(HAVE_GCC43_OR_NEWER OR APPLE)
      if(ENABLE_SSSE3)
	 set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mssse3")
      endif()
      #if(ENABLE_SSE4_1)
      #   set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -msse4.1")
      #endif()
   endif()
endif()

if(X86 OR X86_64)
    if(NOT APPLE)
	if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
	    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mfpmath=both")
	endif()
    endif()
endif()

# Profiling?
if(ENABLE_PROFILING)
    set(EXTRA_EXE_LINKER_FLAGS_RELEASE "${EXTRA_EXE_LINKER_FLAGS_RELEASE} -lprofiler ")
else()
    # Remove unreferenced functions: function level linking
    if(NOT APPLE)
	set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
    endif()
endif()


# High level of warnings.
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wextra -std=c++0x -w") #-w to disable warnings # -Weffc++ -Wextra")

# The -Wno-long-long is required in 64bit systems when including sytem headers.
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
endif()
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
endif()

# set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -DNDEBUG")
set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG} -O0 -DDEBUG -D_DEBUG ")#  -include headers.h -Winvalid-pch")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS} ")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG} ")




if( CMAKE_BUILD_TYPE MATCHES Release )
    message(STATUS "    C++ flags (Release):        ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "    C++ linker flags (Release):       ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
else()    
    message(STATUS "    C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()    


# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin CACHE PATH "Output directory for applications" )

add_subdirectory (src)
