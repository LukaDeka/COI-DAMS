{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    pkgs.uv
    pkgs.python312
    pkgs.stdenv.cc.cc.lib
    pkgs.jdk11
    pkgs.black
  ];

  env = {
    UV_PYTHON_DOWNLOADS = "never";
    UV_PYTHON = "${pkgs.python312}/bin/python3.12";
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    JAVA_HOME = "${pkgs.jdk11}";
  };
}
