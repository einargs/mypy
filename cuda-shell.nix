let pinned-nixpkgs-path = import ./pinned-nixpkgs.nix;
    pinned-pkgs = import pinned-nixpkgs-path {};
in { pkgs ? pinned-pkgs }:

with pkgs;

let
  /* z3-solver = python39Packages.buildPythonPackage rec {
    pname = "z3-solver";
    version = "4.8.14.0";
    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/aa/d6/b46b2fa7966a6183e17aa9ac1ff66d84385cd3e7c9e28d8ef05c6da588f1/z3-solver-4.8.14.0.tar.gz";
      sha256 = "1gmbfgcx9l7yf2qm20dsmpkllabr2r2kyhwpnlxw06g34306mm6y";
    };
    format = "setuptools";
    doCheck = false;
    buildInputs = [ z3 cmake ];
    checkInputs = [];
    nativeBuildInputs = [];
    propagatedBuildInputs = [];
  };
  py = python39.withPackages (ps: [
    z3-solver
  ]); */
  py = python39;
  pyPackages = python39Packages;
in

mkShell {
  buildInputs = [
    z3 ctags cudatoolkit linuxPackages.nvidia_x11 pyPackages.pytorchWithCuda # conda 
  ];

  Z3_LIBRARY_PATH = "${z3.lib}/lib";

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    #conda-shell
    #python3 -m venv .venv
    #source .venv/bin/activate
    #python3 src/play.py
  '';
}
