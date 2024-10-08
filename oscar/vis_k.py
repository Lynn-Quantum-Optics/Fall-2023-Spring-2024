import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter([2**n for n in range(1, 10)], [2*(2**n)-1 for n in range(1, 10)], label='Pisenti et al. 2011', alpha=0.5)
plt.scatter([3], [3], label='Leslie et al. 2019', alpha=0.5)
plt.scatter([2*n for n in range(1, 10)], [2*(2*n)-1 for n in range(1, 10)], label='This work', alpha=0.5)
plt.xlim(0, 20)
plt.ylim(0, 41)
plt.xticks([2*n for n in range(1, 11)])
plt.legend()
plt.xlabel('$d$')
plt.ylabel('$k$')
plt.title('Maximal Distinguishability')
plt.tight_layout()
plt.savefig('k_vs_d.pdf')
plt.show()
