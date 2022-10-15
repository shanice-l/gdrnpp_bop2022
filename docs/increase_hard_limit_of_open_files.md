# How to Increase Open Files Limit in Ubuntu & Debian
Some times we faced issues like
```
Too many open files
```
or
```
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
```

It means our server has hit the max open file limit. This happens due to resource limits set by the system for any user or session.
For example, the max size of files created, the maximum size that may be locked into memory, maximum CPU time used, the maximum number of processes allowed, the maximum size of virtual memory available.

Basically there are two types of limits:

* A hard limit is the maximum allowed limit to a user or session, which is set by the superuser/root.
* A soft limit is the current effective value for the user or session. Which can increase by the user up to the hard limit.

## Check for Current Limits

The `ulimit` command provides control over resources available to each user via a shell. You can use below command to
to get the current settings.
```
ulimit -a
```
To view the current hard limit or soft limit use the following command.
```
ulimit -Sn       # Check soft limit
ulimit -Hn       # Check hard limit
```
## Increase Limit for **Current Session**

Most operating systems can change the open-files limit for the current shell session using the `ulimit -n` command:
```
ulimit -n 200000
```

## Increase **per-user** Limit

You can define per-user open file limit on a Debian based Linux system. To set per-user limit, edit `/etc/security/limits.conf` file in a text editor.
```
sudo vim /etc/security/limits.conf
```
Add the following values in file:
```
* 	 soft     nproc          65535
* 	 hard     nproc          65535
* 	 soft     nofile         500000
* 	 hard     nofile         500000
# jack 	 soft     nproc          200000
# jack 	 hard     nproc          200000
# jack 	 soft     nofile         200000
# jack 	 hard     nofile         200000
```
Here we specifying separate limits which are 200000 for the user `jack` and 65535 will be applied for the rest of the users.
You can change these values per your requirements.

After that enable the pam_limits as followings:
```
sudo vim /etc/pam.d/common-session
```
Add the following line:
```
session required pam_limits.so
```

## Increase **system-wide** Limit

You can also set the limits system-wide by editing the `sysctl` configuration file.

Edit `sysctl.conf` file:
```
vim /etc/sysctl.conf
```

Add the following line:
```
fs.file-max = 2097152
```

Then run the following command to apply the above changes:

```
sysctl -p
```

The above changes will increase the maximum number of files that can remain open system-wide.
The specific user limit can’t be higher than the system-wide limit.

## Important notes

* The change may not take effect in tmux. We need to **close all tmux sessions** and re-open the tmux to get it work.


## Reference
* https://tecadmin.net/increase-open-files-limit-ubuntu/
