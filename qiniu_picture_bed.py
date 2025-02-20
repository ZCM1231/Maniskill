
# -*- coding: utf-8 -*-
# flake8: noqa

from qiniu import Auth, put_file, etag
import qiniu.config
from qiniu import Auth
from qiniu import BucketManager

#需要填写你的 Access Key 和 Secret Key
access_key = 'BziER-3xrxPOXEqYONtP7ZQqEsxKVDWm7ynSh-sF'
secret_key = 'm6FkHnB_iBNfXhZ8jIN1tbpDaPZtZo_e6gXqSXJ9'

#构建鉴权对象
q = Auth(access_key, secret_key)

#要上传的空间
bucket_name = 'vlm-save-space'

#上传后保存的文件名
key = 'rgb_image.png'

#生成上传 Token，可以指定过期时间等
token = q.upload_token(bucket_name, key, 3600)

#要上传文件的本地路径
localfile = '/home/zcm/Pictures/rgb_image_panda.png'

ret, info = put_file(token, key, localfile, version='v2')
print(info)
assert ret['key'] == key
assert ret['hash'] == etag(localfile)





# http://srxiyn0vd.hn-bkt.clouddn.com/rgb_image_panda.png

#初始化Auth状态

q = Auth(access_key, secret_key)

#初始化BucketManager
bucket = BucketManager(q)

#你要测试的空间， 并且这个key在你空间中存在
bucket_name = 'vlm-save-space'
key = 'rgb_image.png'

#删除bucket_name 中的文件 key
ret, info = bucket.delete(bucket_name, key)
print(info)


