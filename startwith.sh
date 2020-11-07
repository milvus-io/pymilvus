#!/bin/sh

if [[ "[ci]aaa" =~ ^"[ci]".* ]]; then
    echo "yes"
else
    echo "no"
fi
