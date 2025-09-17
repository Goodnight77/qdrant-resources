### Best Practices for Production:
1. **Indexing**: Use HNSW for fast approximate search on large datasets
2. **Memory**: Consider quantization to reduce memory usage
3. **Storage**: Use on-disk storage for large collections
4. **Filtering**: Create payload indexes for frequently filtered fields
5. **Monitoring**: Track collection size, search latency, and memory usage
6. **Updates**: Use batch operations for bulk inserts/updates
7. **Optimization**: Tune HNSW parameters (m, ef_construct) for your use case
