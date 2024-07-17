mkdir ~/dotfiles
cd ~/dotfiles
git init
cp ~/.bashrc .
git add .bashrc
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/dotfiles.git
git push -u origin main
