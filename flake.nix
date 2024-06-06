{
  description = "snax devshell";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
          conf_basebath = "/home/anton/.config/pycharm/snax";
        in
          {
            devShells.default = with pkgs; mkShell {
              LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";
              buildInputs = [
                nodejs_22 ruff black isort llvmPackages_17.mlir
              ];
	      name = "snax-nix";
              JAVA_TOOL_OPTIONS = "-Didea.config.path=${conf_basebath}/conf -Didea.system.path=${conf_basebath}/sys -Didea.plugins.path=${conf_basebath}/plugins -Didea.log.path=${conf_basebath}/sys/logs";
            };
          }
    );
}
