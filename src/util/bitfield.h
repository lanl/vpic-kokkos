#ifndef BitField_h
#define BitField_h

class BitField
	{
	public:
		
		BitField(uint64_t setbits = UINT64_MAX) : bits_(setbits) {}
		BitField(const BitField & bf) : bits_(bf.bits_) {}
		~BitField() {}

		/*!---------------------------------------------------------------------
		 * Set bits in mask.
		----------------------------------------------------------------------*/
		uint64_t set(uint64_t mask) {
			return bits_ |= mask;
		} // set

		/*!---------------------------------------------------------------------
		 * Set individual bit.
		----------------------------------------------------------------------*/
		uint64_t setibit(size_t bit) {
			uint64_t tmp = 1ULL<<bit;
			return bits_ |= tmp;
		} // setibit

		/*!---------------------------------------------------------------------
		 * Clear bits in mask.
		----------------------------------------------------------------------*/
		uint64_t clear(uint64_t mask) {
			return bits_ &= ~mask;
		} // clear

		/*!---------------------------------------------------------------------
		 * Clear individual bit.
		----------------------------------------------------------------------*/
		uint64_t clearbit(size_t bit) {
			uint64_t tmp = 1ULL<<bit;
			return bits_ &= ~tmp;
		} // clearbit

		/*!---------------------------------------------------------------------
		 * Check whether bit at index is set.
		----------------------------------------------------------------------*/
		bool bitset(size_t bit) const {
			uint64_t tmp = 1ULL<<bit;
			return tmp & bits_;
		} // bitset

		/*!---------------------------------------------------------------------
		 * Check whether bits in mask are set.
		----------------------------------------------------------------------*/
		bool bitsset(uint64_t mask) const {
			uint64_t tmp = 1ULL<<mask;
			return tmp & bits_;
		} // bitset

		/*!---------------------------------------------------------------------
		 * Check whether bit at index is clear.
		----------------------------------------------------------------------*/
		bool bitclear(size_t bit) const {
			uint64_t tmp = 1ULL<<bit;
			return !(tmp & bits_);
		} // bitclear

		/*!---------------------------------------------------------------------
		 * Return the sum of the set bits in the mask.
		----------------------------------------------------------------------*/
		size_t bitsum(const size_t * indeces, size_t size) const {
			size_t sum(0);
			for(size_t i(0); i<size; i++) {
				if(bitset(indeces[i])) {
					++sum;
				} // if
			} // for
			return sum;
		} // bitclear

		/*!---------------------------------------------------------------------
		 * Return the sum of the set bits.
		----------------------------------------------------------------------*/
		size_t bitsum() const {
			size_t sum(0);
			for(size_t i(0); i<64; i++) {
				if(bitset(i)) {
					++sum;
				} // if
			} // for
			return sum;
		} // bitclear

	private:

		uint64_t bits_;

	}; // class BitField

#endif // BitField_h
