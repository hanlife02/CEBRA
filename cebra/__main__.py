#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""CEBRA命令行界面。


"""

import argparse
import sys

import cebra


def train(parser, kwargs):
    """训练一个新的CEBRA模型，可能从检查点开始。"""
    parser.add_argument("--variant", choices=cebra.CEBRA.get_variants())
    args, kwargs = parser.parse_known_args()
    cebra_cls = cebra.CEBRA.get_variant(args.variant)

    cebra_cls.add_arguments(parser)
    args, kwargs = parser.parse_known_args(kwargs)
    experiment = cebra_cls.from_args(args=args)

    parser.add_argument(
        "--override",
        "-r",
        action="store_true",
        help="覆盖现有检查点（不加载）。",
    )
    args, kwargs = parser.parse_known_args(kwargs)

    if not args.override:
        experiment.load()
    try:
        experiment.train()
    except KeyboardInterrupt:
        print("训练已中止。正在保存模型。")
        sys.exit(1)
    except Exception as exception:
        print("发生错误。正在中止训练。")
        raise exception
    finally:
        experiment.save()


def transform(parser, kwargs):
    """使用训练好的CEBRA模型转换现有数据集。"""
    print("transform a dataset.")


def app(parser, kwargs):
    """启动服务器以提供CEBRA的Web界面。"""
    from cebra.integrations.streamlit import App

    App.add_arguments(parser)
    args = parser.parse_args(kwargs)
    App.run(args)


def main():
    parser = argparse.ArgumentParser(
        "cebra", formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    commands = {"train": train, "transform": transform, "app": app}
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "command",
        choices=list(commands.keys()),
        help="要运行的子命令：\n" +
        "\n".join(f"{name}\t{cmd.__doc__}" for name, cmd in commands.items()),
        metavar="command",
    )
    args, kwargs = parser.parse_known_args()
    if args.version:
        print(f"CEBRA {cebra.__version__}.")
        sys.exit(0)
    command = commands.get(args.command)

    parser = argparse.ArgumentParser(f"cebra {args.command}")
    command(parser, kwargs)


if __name__ == "__main__":
    main()
