#!/usr/bin/env bash

export COUPLING_HOME

if [[ -f "$COUPLING_HOME/.env" ]]; then
  set -a
  source "$COUPLING_HOME/.env"
  set +a
fi
