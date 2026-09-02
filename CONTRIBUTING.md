# Contributing to B-HGP (Binary Hexagonal Galois Pattern)

We welcome contributions to the B-HGP project! Whether you're reporting bugs, suggesting improvements, or submitting code changes, please follow these guidelines.

## How to Contribute

### 1. Reporting Issues

If you've found a bug or have a feature request:

1. Check existing [issues](https://github.com/gkis-conda/robust_calibration_pattern/issues) to avoid duplicates
2. Create a new issue with:
   - Clear description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Your environment (Python version, OS, dependencies)

### 2. Code Contributions

To contribute code changes:

1. **Fork** the repository
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** using the existing test suite:
   ```bash
   python pattern_decoder_test.py -p ./results --save-images
   ```
5. **Submit a Pull Request** with:
   - Description of changes and motivation
   - Reference to related issues (if any)
   - Confirmation that tests pass

### 3. Development Setup

To set up a development environment:

```bash
git clone https://github.com/gkis-conda/robust_calibration_pattern.git
cd robust_calibration_pattern
pip install -r requirements.txt
```

### 4. Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include type hints where practical
- Keep functions focused and testable
- Update documentation for new features

### 5. Testing

Always run tests before submitting:

```bash
# Run verification suite
python pattern_decoder_test.py -p ./test_results --save-images

# Run single image extraction
python detector.py --input <test_image.png>

# Run Blender benchmarks (if available)
blender --background --python blender_benchmark.py -- --engine hgp --p <result_folder>
```

All tests should pass with zero failures.

### 6. Documentation

- Update README.md if you add features or change behavior
- Add comments for complex algorithms
- Include usage examples for new functionality
- Document any new command-line arguments

## Citation

If you use B-HGP in your research, please cite this work using the paper.bib reference provided in the repository.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help other contributors succeed
- Report problematic behavior to the maintainers

## Contact

For questions or discussions:
- Open an issue on GitHub
- Reference the research paper submitted to IJCV for scientific context
- Contact: gennadiykiss@gmail.com for commercial inquiries

Thank you for contributing to B-HGP! 🎉
