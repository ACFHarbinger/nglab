import React from 'react';

const Icon = (props: any) => <svg {...props} />;

module.exports = new Proxy({}, {
    get: () => Icon
});
