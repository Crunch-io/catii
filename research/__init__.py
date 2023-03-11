"""Run these on a completely unloaded system (turn off all other apps), and you
probably also need:

sudo swapoff -a
sudo sysctl vm.overcommit_memory=2

to avoid slowness and lockups. Then run these when finished:

sudo swapon -a
sudo sysctl vm.overcommit_memory=0
"""
