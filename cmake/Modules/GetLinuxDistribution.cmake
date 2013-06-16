# Gets the linux distribution
#
# This module defines:
# LINUX_DISTRIBUTION_IS_REDHAT
# LINUX_DISTRIBUTION_IS_GENTOO

# Check: Can /usr/bin/lsb_release -a be used?

set(LINUX_DISTRIBUTION_IS_REDHAT FALSE)
set(LINUX_DISTRIBUTION_IS_GENTOO FALSE)

if (UNIX)
    if (EXISTS "/etc/redhat-release")
        set(LINUX_DISTRIBUTION_IS_REDHAT TRUE)
    endif()

    if (EXISTS "/etc/gentoo-release")
        set(LINUX_DISTRIBUTION_IS_GENTOO TRUE)
    endif()
endif()