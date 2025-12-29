#!/bin/bash

# Cleanup script for GitHub repository
# Moves unnecessary files to ../PDF_ARCHIVE to keep repo clean

echo "🧹 Cleaning up repository for GitHub..."

# Create archive directory outside the repo
ARCHIVE_DIR="../PDF_ARCHIVE"
mkdir -p "$ARCHIVE_DIR"

# Move large unnecessary directories
echo "📦 Moving large directories to archive..."
mv models_backup "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved models_backup/ (5.4GB)"
mv logs "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved logs/"
mv test_results "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved test_results/"
mv catboost_info "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved catboost_info/"
mv analysis "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved analysis/"
mv evaluation "$ARCHIVE_DIR/" 2>/dev/null && echo "  ✓ Moved evaluation/"

# Move historical VM data
echo "📦 Moving historical VM data..."
mkdir -p "$ARCHIVE_DIR/data"
mv data/vm_collected "$ARCHIVE_DIR/data/" 2>/dev/null && echo "  ✓ Moved data/vm_collected/"
mv data/vm_processed/archived "$ARCHIVE_DIR/data/" 2>/dev/null && echo "  ✓ Moved data/vm_processed/archived/"

# Move documentation clutter to archive
echo "📄 Moving documentation clutter..."
mkdir -p "$ARCHIVE_DIR/docs_archive"

# List of docs to archive (strategy/analysis docs not needed for operations)
DOCS_TO_ARCHIVE=(
    "ARCHITECTURE.md"
    "AUDIT_REPORT.md"
    "CHANGES_LLM_REQUIRED.md"
    "DAILY_BATCH_CICD_GUIDE.md"
    "DAILY_PIPELINE_DESIGN.md"
    "DATA_DIVERSITY_STRATEGY.md"
    "DEPLOYMENT_SUMMARY.md"
    "DIFFICULT_URLS_TEST_RESULTS.md"
    "ENSEMBLE_LLM_INTEGRATION.md"
    "FEATURE_VALIDATION.md"
    "FEATURE_VALIDATION_FINDINGS.md"
    "FILE_DEPENDENCY_ANALYSIS.md"
    "HOLISTIC_DIVERSITY_STRATEGY.md"
    "HOLISTIC_IMPLEMENTATION_GUIDE.md"
    "INCREMENTAL_DIVERSITY_STRATEGY.md"
    "META_LEARNING_RESULTS.md"
    "MISSING_PATTERNS_AUDIT.md"
    "MODEL_FAMILY_SELECTION.md"
    "OPTIMAL_CICD_STRATEGY.md"
    "ORCHESTRATION_GUIDE.md"
    "PARALLEL_FEATURE_EXTRACTION.md"
    "PIPELINE_EXECUTION_GUIDE.md"
    "PIPELINE_STATUS.md"
    "PIPELINE_WORKING.md"
    "PROJECT_SUMMARY.md"
    "REPOSITORY_ANALYSIS.md"
    "VALIDATION_SUMMARY.md"
    "VM_VS_LOCAL_ANALYSIS.md"
    "README_PIPELINE.md"
)

for doc in "${DOCS_TO_ARCHIVE[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" "$ARCHIVE_DIR/docs_archive/" && echo "  ✓ Moved $doc"
    fi
done

# Move test/validation scripts (not needed in production)
echo "🔬 Moving test/validation scripts..."
mkdir -p "$ARCHIVE_DIR/scripts_archive"
mv test_difficult_urls.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved test_difficult_urls.py"
mv test_llm_explanations.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved test_llm_explanations.py"
mv validate_features.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved validate_features.py"
mv quick_validation_summary.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved quick_validation_summary.py"
mv fix_labels.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved fix_labels.py"
mv meta_learning_ensemble_weights.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved meta_learning_ensemble_weights.py"
mv meta_learning_simple.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved meta_learning_simple.py"
mv meta_learning_results.json "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved meta_learning_results.json"
mv pipeline.py "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved pipeline.py"
mv cleanup.sh "$ARCHIVE_DIR/scripts_archive/" 2>/dev/null && echo "  ✓ Moved cleanup.sh"

# Remove pytest config (not using pytest in CI)
rm pytest.ini 2>/dev/null && echo "  ✓ Removed pytest.ini"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "  Archive location: $ARCHIVE_DIR"
echo "  Repository is now clean and ready for GitHub"
echo ""
echo "🔍 Current repo size:"
du -sh . 2>/dev/null
