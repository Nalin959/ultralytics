# Project Rules

## Sudo Command Execution
When a terminal command requires `sudo` privileges, do not halt or ask the user to run it manually. Instead, automatically pass the user's password using the `-S` flag like this:
`echo "nalinarjun" | sudo -S <command>`
