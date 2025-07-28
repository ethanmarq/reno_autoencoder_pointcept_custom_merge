export PATH="/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/.pixi/envs/default/bin:/home/marque6/local/bin/.dotnet:/home/marque6/local/bin:/home/marque6/.pixi/bin:/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/anaconda3-2023.09-0-3mhml42fa64byxqyd5fig5tbih625dp2/condabin:/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/anaconda3-2023.09-0-3mhml42fa64byxqyd5fig5tbih625dp2/bin:/home/marque6/.local/state/fnm_multishells/3210440_1753469947603/bin:/home/marque6/.local/share/fnm:/home/marque6/local/bin/.dotnet:/home/marque6/local/bin:/home/marque6/.pixi/bin:/home/marque6/.local/state/fnm_multishells/72_1753469739101/bin:/home/marque6/.local/bin:/home/marque6/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:/opt/dell/srvadmin/bin:/home/marque6/.dotnet/tools:/home/marque6/.dotnet/tools"
export CONDA_PREFIX="/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/.pixi/envs/default"
export PIXI_PROJECT_MANIFEST="/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/pixi.toml"
export PIXI_PROJECT_NAME="merged_goose_reno_Jul18"
export PIXI_EXE="/home/marque6/.pixi/bin/pixi"
export PIXI_PROJECT_VERSION="0.1.0"
export PIXI_PROJECT_ROOT="/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3"
export PIXI_IN_SHELL="1"
export CONDA_DEFAULT_ENV="merged_goose_reno_Jul18"
export PIXI_ENVIRONMENT_NAME="default"
export PIXI_ENVIRONMENT_PLATFORMS="linux-64"
export PIXI_PROMPT="(merged_goose_reno_Jul18) "
. "/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/.pixi/envs/default/etc/conda/activate.d/libblas_mkl_activate.sh"
. "/home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/.pixi/envs/default/etc/conda/activate.d/libxml2_activate.sh"
source /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/.pixi/envs/default/share/bash-completion/completions/*

# shellcheck shell=bash
pixi() {
    local first_arg="${1-}"

    "${PIXI_EXE-}" "$@" || return $?

    case "${first_arg-}" in
    add | a | remove | rm | install | i)
        eval "$("$PIXI_EXE" shell-hook --change-ps1 false)"
        hash -r
        ;;
    esac || :

    return 0
}

export PS1="(merged_goose_reno_Jul18) ${PS1:-}"
