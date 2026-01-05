#!/usr/bin/env python3
"""
批量处理email的实验脚本
支持选择数据集、操作类型和模型，批量处理并记录结果
"""

import json
import os
import argparse
from datetime import datetime
from generate import GenerateEmail
from dotenv import load_dotenv

load_dotenv()


def load_emails_from_jsonl(path):
    """从jsonl文件加载email数据"""
    emails = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                emails.append(json.loads(line))
    return emails


def parse_review_json(review_str):
    """解析评审JSON字符串，提取rating和explanation"""
    try:
        review_dict = json.loads(review_str)
        return review_dict.get("rating"), review_dict.get("explanation", "")
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，尝试查找rating数字
        import re
        rating_match = re.search(r'"rating":\s*(\d+)', review_str)
        rating = int(rating_match.group(1)) if rating_match else None
        return rating, review_str


def process_email_batch(
    dataset_path: str,
    operation: str,
    model: str,
    output_log_path: str = "logs/batch_experiments_edge.jsonl",
):
    """
    批量处理数据集中的email
    
    Args:
        dataset_path: 数据集文件路径
        operation: 操作类型 (elaborate, tone, shorten)
        model: 模型名称
        output_log_path: 输出日志文件路径
    """
    # 验证操作类型
    valid_operations = ["elaborate", "tone", "shorten"]
    if operation not in valid_operations:
        raise ValueError(f"操作类型必须是以下之一: {valid_operations}")

    # 检查数据集文件是否存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

    # 创建日志目录
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

    # 初始化生成器
    print(f"初始化模型: {model}")
    generator = GenerateEmail(model=model)

    # 加载数据集
    print(f"加载数据集: {dataset_path}")
    emails = load_emails_from_jsonl(dataset_path)
    print(f"找到 {len(emails)} 条email记录")

    # 处理每条email
    processed_count = 0
    error_count = 0

    for idx, email in enumerate(emails, 1):
        email_id = email.get("id", f"unknown_{idx}")
        email_content = email.get("content", "")

        print(f"\n[{idx}/{len(emails)}] 处理 Email ID: {email_id}")

        try:
            # 生成处理后的email
            print(f"  执行操作: {operation}")
            edited_email = generator.generate(
                operation,
                selected_text=email_content,
            )

            # 获取faithfulness评审
            print(f"  获取faithfulness评审...")
            faithfulness_review = generator.generate(
                "faithfulness_judge",
                selected_text=email_content,
                model_response=edited_email,
            )

            # 获取completeness评审
            print(f"  获取completeness评审...")
            completeness_review = generator.generate(
                "completeness_check",
                selected_text=email_content,
                model_response=edited_email,
            )

            # 解析评审分数
            faithfulness_rating, faithfulness_explanation = parse_review_json(
                faithfulness_review
            )
            completeness_rating, completeness_explanation = parse_review_json(
                completeness_review
            )

            # 构建日志记录
            log_record = {
                "model": model,
                "operation": operation,
                "dataset_path": os.path.abspath(dataset_path),
                "email_id": email_id,
                "email_content": email_content,
                "output_content": edited_email,
                "faithfulness_rating": faithfulness_rating,
                "faithfulness_explanation": faithfulness_explanation,
                "completeness_rating": completeness_rating,
                "completeness_explanation": completeness_explanation,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # 写入日志文件
            with open(output_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + "\n")

            processed_count += 1
            print(f"  ✓ 完成 (Faithfulness: {faithfulness_rating}, Completeness: {completeness_rating})")

        except Exception as e:
            error_count += 1
            print(f"  ✗ 错误: {str(e)}")
            # 记录错误信息
            error_record = {
                "model": model,
                "operation": operation,
                "dataset_path": os.path.abspath(dataset_path),
                "email_id": email_id,
                "email_content": email_content,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(output_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"  成功: {processed_count}")
    print(f"  失败: {error_count}")
    print(f"  总计: {len(emails)}")
    print(f"  日志文件: {output_log_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="批量处理email实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认模型处理shorten操作
  python batch_experiment.py datasets/shorten.jsonl shorten

  # 指定模型
  python batch_experiment.py datasets/shorten.jsonl shorten --model gpt-4

  # 指定输出日志文件
  python batch_experiment.py datasets/tone.jsonl tone --output logs/my_experiment.jsonl
        """,
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="数据集文件路径 (例如: datasets/shorten.jsonl)",
    )

    parser.add_argument(
        "operation",
        type=str,
        choices=["elaborate", "tone", "shorten"],
        help="操作类型: elaborate, tone, 或 shorten",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型名称 (默认使用环境变量 DEPLOYMENT_NAME)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="logs/batch_experiments.jsonl",
        help="输出日志文件路径 (默认: logs/batch_experiments.jsonl)",
    )

    args = parser.parse_args()

    # 确定使用的模型
    model = args.model or os.getenv("DEPLOYMENT_NAME")
    if not model:
        raise ValueError(
            "未指定模型。请使用 --model 参数或设置环境变量 DEPLOYMENT_NAME"
        )

    # 执行批量处理
    process_email_batch(
        dataset_path=args.dataset,
        operation=args.operation,
        model=model,
        output_log_path=args.output,
    )


if __name__ == "__main__":
    main()
