set(HAVE_SWIG FALSE)
if(USE_SWIG)
  # Search for SWIG, Python, and NumPy
  find_package(SWIG)
  find_package(PythonLibs)
  set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  find_package(NumPy)
  if(SWIG_FOUND AND PYTHONLIBS_FOUND AND NUMPY_FOUND)
    if(${NUMPY_VERSION_MAJOR} GREATER 0 AND ${NUMPY_VERSION_MINOR} GREATER 6)
      include_directories(${NUMPY_INCLUDE_DIRS})
      set(HAVE_SWIG TRUE)
    else()
      message(STATUS "NumPy version must be >= 1.7")
    endif()
  else()
    message(STATUS "Did not find necessary SWIG, Python, and NumPy libs")
  endif()
endif()
