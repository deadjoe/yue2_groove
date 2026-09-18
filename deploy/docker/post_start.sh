#!/bin/bash
# Called by RunPod's /start.sh after sshd/jupyter setup (a no-op on plain Docker). The app is
# started in the background so /start.sh can keep the container alive; logs go to the
# container's stdout.
groove-start &
