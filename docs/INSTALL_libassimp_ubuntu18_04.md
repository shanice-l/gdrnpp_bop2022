# Installation of libassimp-dev 3.2 in Ubuntu 18.04

* Check assimp version:
    ```
    dpkg -l |grep assimp
    ```

* Remove libassimp-dev if it is installed previously
    ```
    sudo apt remove libassimp-dev
    sudo apt remove libassimp5*  # or 4*
    ```

* Install libassimp3v5=3.2~dfsg-3 via [source](https://www.ubuntuupdates.org/package/core/xenial/universe/base/libassimp3v5)
    * Download the debian file (.deb) corresponding to your system
        ```
        wget http://security.ubuntu.com/ubuntu/pool/universe/a/assimp/libassimp3v5_3.2~dfsg-3_amd64.deb
        ```
    * Run command `sudo dpkg -i libassimp3v5_3.2_*.deb` (need to specify the file you download)
        ```
        sudo dpkg -i libassimp3v5_3.2~dfsg-3_amd64.deb
        ```

* Install libassimp-dev=3.2~dfsg-3 via [source](https://launchpad.net/ubuntu/xenial/+package/libassimp-dev)
    ```
    wget http://launchpadlibrarian.net/230132835/libassimp-dev_3.2~dfsg-3_amd64.deb
    sudo dpkg -i libassimp-dev_3.2~dfsg-3_amd64.deb
    ```

* Check assimp version again:
    ```
    dpkg -l |grep assimp
    ```
