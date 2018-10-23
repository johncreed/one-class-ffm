#! /bin/bash

git add logs/
git commit -m "Update log"
git pull origin track_logs
git push origin track_logs
