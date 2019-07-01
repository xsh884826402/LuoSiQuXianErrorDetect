def search(nums, target: int) :
    length = len(nums)
    l, l2 = 0, 0
    r, r2 = length - 1, length - 1
    while (l <= r):
        m = int((l + r) / 2)
        if target == nums[m]:
            return m
        elif nums[m] >= nums[l]:
            if target >= nums[l] and target <= nums[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if target >= nums[m] and target <= nums[r]:
                l = m + 1
            else:
                r = m - 1
input = [9, 10, 12, 24, 34, 2, 3, 6]
target = 24
print(search(input,target))

