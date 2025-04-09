#!/binbash


SESSION="impact"
SESSIONEXISTS=$(tmux list-sessions | grep $SESSION)


if [ "$SESSIONEXISTS" = "" ]
then

    # start new session
    tmux new-session -d -s $SESSION


    # first pane
    tmux rename-window -t 0 "main"
    tmux send-keys -t $SESSION:0 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:0 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:0 "clear" C-m


    # second pane
    tmux new-window -t $SESSION:1 -n "foxglove"
    tmux send-keys -t $SESSION:1 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:1 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:1 "clear" C-m
    tmux send-keys -t $SESSION:1 "ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=127.0.0.1" C-m


    # third pane
    tmux new-window -t $SESSION:2 -n "ros2-workbench"
    tmux split-window -t $SESSION:2 -h
    tmux split-window -t $SESSION:2 -v
    tmux split-window -t $SESSION:2.0 -v

    tmux select-pane -t $SESSION:2.0
    tmux send-keys -t $SESSION:2.0 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:2.0 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:2.0 "clear" C-m

    tmux select-pane -t $SESSION:2.1
    tmux send-keys -t $SESSION:2.1 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:2.1 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:2.1 "clear" C-m

    tmux select-pane -t $SESSION:2.2
    tmux send-keys -t $SESSION:2.2 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:2.2 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:2.2 "clear" C-m

    tmux select-pane -t $SESSION:2.3
    tmux send-keys -t $SESSION:2.3 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:2.3 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:2.3 "clear" C-m
    tmux send-keys -t $SESSION:2.3 "htop" C-m


    # fourth pane
    tmux new-window -t $SESSION:3 -n "bash"
    tmux send-keys -t $SESSION:3 "micromamba activate impact" C-m
    tmux send-keys -t $SESSION:3 "source install/setup.bash" C-m
    tmux send-keys -t $SESSION:3 "clear" C-m


fi

tmux attach-session -t $SESSION:0
