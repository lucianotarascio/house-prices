# ---------------------------------------------------------
# 8. Exportar Archivo para Submission de Kaggle
# ---------------------------------------------------------
submission = pd.DataFrame({"Id": test_ids, "SalePrice": final_preds})

submission.to_csv("submission.csv", index=False)
print("¡Archivo 'submission.csv' generado exitosamente!")
