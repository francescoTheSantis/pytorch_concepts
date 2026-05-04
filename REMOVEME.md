Here's what you need to do step by step:

## 1. Add the original repo as `upstream` (if not done yet)
```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/REPO.git
git fetch upstream
```

## 2. Fetch the existing PR's branch
You need to get the branch the existing PR is based on. You can find the branch name on the GitHub PR page.

```bash
git fetch upstream pull/PR_NUMBER/head:existing-pr-branch
# Or if the contributor's fork is accessible:
git fetch upstream <existing-pr-branch-name>
```

Then check it out:
```bash
git checkout existing-pr-branch
```

## 3. Bring in your local changes
You have two options depending on your situation:

**Option A — Cherry-pick** (if your changes are in specific commits):
```bash
git cherry-pick <your-commit-hash>
# Or a range:
git cherry-pick <first-hash>^..<last-hash>
```

**Option B — Merge your branch into the PR branch** (if you want all your changes at once):
```bash
git merge your-local-branch
```

**Option C — If you just have unstaged/staged changes** (not yet committed):
```bash
# Stash them first on your current branch
git stash

# Switch to the PR branch
git checkout existing-pr-branch

# Apply the stash
git stash pop
```

## 4. Push the changes to the right remote
```bash
# If you have write access to the PR's branch directly:
git push upstream existing-pr-branch

# If you need to push to your fork first:
git push origin existing-pr-branch
```
Then on GitHub, you can update the PR's base branch if needed.

---

**Key tip:** Always check the PR page on GitHub to confirm the exact **branch name** and **target repo** (original vs. a fork) before pushing, so you don't accidentally create a new PR instead of updating the existing one.