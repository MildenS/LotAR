from conan import ConanFile
from conan.tools.cmake import cmake_layout


class LotAR(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.8.1")
        self.requires("asio/1.30.1")

    def layout(self):
        cmake_layout(self)