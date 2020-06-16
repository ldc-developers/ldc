# Gets the linux distribution
#
# This module defines:
# LINUX_DISTRIBUTION_IS_GENTOO

# Check: Can /usr/bin/lsb_release -a be used?

set(LINUX_DISTRIBUTION_IS_GENTOO FALSE)

if (UNIX)
    if (EXISTS "/etc/gentoo-release")
        set(LINUX_DISTRIBUTION_IS_GENTOO TRUE)
    endif()
endif()
