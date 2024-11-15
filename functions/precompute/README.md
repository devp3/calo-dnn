# precompute

The purpose of this subdirectory is to list any function that is needed to compute a variable listed in the config file that is not a member of the original branches loaded from the intial root file. For instance, if we want to include a variable 'el1_sinh_eta12' in the feature set, we would need to compute this variable from 'el1_etas1' and 'el1_etas2' following the formula: 