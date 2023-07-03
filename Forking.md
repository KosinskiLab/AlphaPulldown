#Fork from github to gitlab
#local clone with submodules
git clone --recurse-submodules git@github.com:KosinskiLab/AlphaPulldown.git
cd AlphaPulldown 
git submodule init
git submodule update 

#add gitlab as a remote
git remote add gitlab https://git.embl.de/grp-kosinski/AlphaPulldown.git
Check that it’s added by:
git remote -v
you should get smth like:
gitlab	https://git.embl.de/grp-kosinski/AlphaPulldown.git (fetch)
gitlab	https://git.embl.de/grp-kosinski/AlphaPulldown.git (push)
origin	git@github.com:KosinskiLab/AlphaPulldown.git (fetch)
origin	git@github.com:KosinskiLab/AlphaPulldown.git (push)

#set upstream to gitlab
#Note: that you need to change upstream to github each time you want to rebase
#See the deatils below
git fetch gitlab
git branch --set-upstream-to=gitlab/master
git pull gitlab/master
#main is sometimes called master on gitlab
git push gitlab HEAD:master
#update submodules from remote
git submodule update –remote


#Rebase
git remote add upstream https://github.com/KosinskiLab/AlphaPulldown.git   
#add GitHub repository to remotes
git fetch upstream   # fetch the latest changes from the remote repository
git branch -u upstream/main   # set the new upstream branch to main
git pull --rebase   # rebase your current branch to synchronize with the new upstream branch

# If there are conflicts, resolve them using a merge tool
git mergetool

# Once conflicts are resolved, continue the rebase
git rebase --continue

# Switch the pointer back to the GitLab repository
git remote set-url origin https://git.embl.de/grp-kosinski/AlphaPulldown.git

# Commit the changes
git commit -m "Rebased from github/main"

# Push the changes to the GitLab repository with the --force option
git push --force origin [name-of-your-branch]

