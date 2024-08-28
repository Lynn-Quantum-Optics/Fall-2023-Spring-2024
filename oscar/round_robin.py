# code to generate round-robin pairings for a given number of participants
# MIT license
# Author: Oscar Scholin

def round_robin_pairings(d):
    # generate list of participants
    participants = list(range(d))
    
    # store pairings
    pairings = []

    # number of rounds played is d-1
    for _ in range(d - 1):
        round_pairings = []
        
        # pair participant 0 with the participant in last position
        round_pairings.append((participants[0], participants[-1]))
        
        # pair remaining participants
        for i in range(1, d // 2):
            round_pairings.append((participants[i], participants[-i-1]))

        # add current round pairings to list of all pairings
        pairings.append(round_pairings)

        # rotate participants except fixed participant 0
        participants = [participants[0]] + [participants[-1]] + participants[1:-1]

    return pairings


if __name__ == "__main__":
    # example for d = 6 participants
    d = 6
    pairings = round_robin_pairings(d)
    for round_number, round_pair in enumerate(pairings):
        print(f"Round {round_number + 1}: {round_pair}")

    # ensure that all participants are paired with all other participants
    total_pairs = set(sum(pairings, []))
    print(f"Total pairs: {total_pairs},", "Number of pairs:", len(total_pairs))
    assert len(total_pairs) == d * (d - 1) // 2

