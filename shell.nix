let pinned-nixpkgs-path = import ./pinned-nixpkgs.nix;
    pinned-pkgs = import pinned-nixpkgs-path {};
in { pkgs ? pinned-pkgs }:

with pkgs;

let
  py = python39;
  pyPackages = python39Packages;
in

mkShell {
  buildInputs = [
    z3 ctags py gcc
  ];

  Z3_LIBRARY_PATH = "${z3.lib}/lib";
  LD_LIBRARY_PATH = "${lib.makeLibraryPath [stdenv.cc.cc]}";

  shellHook = ''
    python3 -m venv ./env
    source ./env/bin/activate
    #python3 src/play.py
  '';
}
