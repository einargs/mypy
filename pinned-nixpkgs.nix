let
   hostPkgs = import <nixpkgs> {};
   # Generated through:
   # To get the latest revision, just go to:
   # https://nixos.org/channels/nixos-unstable/git-revision
   # nix-shell -p nix-prefetch-git --run "nix-prefetch-git  https://github.com/nixos/nixpkgs.git --rev {rev} > nixpkgs-version.json"
   pinnedVersion = hostPkgs.lib.importJSON ./nixpkgs-version.json;

   pinnedPkgs = hostPkgs.fetchFromGitHub {
     owner = "NixOS";
     repo = "nixpkgs-channels";
     inherit (pinnedVersion) rev sha256;
   };
 in pinnedPkgs
