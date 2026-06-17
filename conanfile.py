from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class Fusion4DConan(ConanFile):
    name = "fusion4d"
    version = "0.1"
    package_type = "library"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_demo": [True, False],
        "with_visual_feature_matching": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_demo": True,
        "with_visual_feature_matching": False,
        "opencv/*:with_cuda": True,
        "opencv/*:dnn": False,
        "opencv/*:gapi": False,
        "opencv/*:ml": False,
        "opencv/*:photo": False,
        "opencv/*:video": True,
        "opencv/*:with_ffmpeg": False,
        "freeglut/*:with_wayland": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "ConfigFileExamples/*",
        "DeformableFusion/*",
        "McLib2/*",
        "include/*",
        "LICENSE",
        "Readme.md",
        "src/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("Fusion4D Linux packaging is currently supported only on Linux.")
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, "17")

    def requirements(self):
        self.requires("opencv/4.13.0")
        self.requires("jsoncpp/1.9.6")
        self.requires("zlib/1.3@camposs/stable")
        self.requires("vxl/1.18.0@camposs/stable")
        self.requires("suitesparse/7.9.0@camposs/stable")
        self.requires("ceres-solver/2.2.0@camposs/stable")
        if self.options.with_demo:
            self.requires("freeglut/3.8.0")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.24 <4]")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["FUSION4D_ENABLE_CONAN"] = True
        tc.variables["FUSION4D_WITH_DEMO"] = self.options.with_demo
        tc.variables["FUSION4D_WITH_VISUAL_FEATURE_MATCHING"] = self.options.with_visual_feature_matching
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder, dst=f"{self.package_folder}/licenses")
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Fusion4D")
        self.cpp_info.set_property("cmake_target_name", "Fusion4D::Fusion4DCore")
        self.cpp_info.bindirs = ["bin"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.libs = [
            "Fusion4DCore",
            "FastNonrigidMatching",
            "NonRigidMatch",
            "Basics",
            "CSurface",
            "CameraView",
            "Utility",
            "LibUtility",
        ]
