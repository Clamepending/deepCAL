    #!/bin/bash
    # Job name:
    #SBATCH --job-name=hello_world
    #
    # Account:
    #SBATCH --account=fc_contact
    #
    # Partition:
    #SBATCH --partition=savio3
    #
    # Wall clock limit:
    #SBATCH --time=00:00:30
    #
    ## Command(s) to run:
    echo "hello world"